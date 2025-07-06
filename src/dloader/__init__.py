from __future__ import annotations

import asyncio
from collections.abc import Hashable, Iterable, Mapping, MutableMapping, Sequence
from types import TracebackType
from typing import Generic, Protocol, Self, TypeVar

__all__ = (
    "DataLoader",
    "LoadFunction",
)

_K = TypeVar("_K", bound=Hashable, contravariant=True)
_V = TypeVar("_V", covariant=True)


class LoadFunction(Protocol, Generic[_K, _V]):
    async def __call__(self, keys: Sequence[_K], /) -> Sequence[_V | Exception]: ...


_KeyType = TypeVar("_KeyType", bound=Hashable)
_ResultType = TypeVar("_ResultType")


class DataLoader(Generic[_KeyType, _ResultType]):
    """
    Batches and caches asynchronous data loading operations.

    Solves the N+1 query problem by collecting multiple load requests and executing them
    as a single batch operation. Deduplicates requests within a batch and caches results
    to avoid redundant loads.

    The load function receives a sequence of keys and must return results in the same
    order. Results can be values or Exception instances which are propagated to the
    appropriate futures.

    All asyncio tasks are tracked internally and cleaned up during shutdown, preventing
    task leaks. Use as an async context manager for automatic cleanup or call shutdown()
    manually.

    Example usage
    -------------

    >>> _batches = list[Sequence[int]]()
    >>> async def load_users(user_ids: Sequence[int]) -> Sequence[str]:
    ...     _batches.append(user_ids)
    ...     return [f"user_{id}" for id in user_ids]
    >>>
    >>> async def example():
    ...     async with DataLoader(load_users) as loader:
    ...         results = await asyncio.gather(
    ...             loader.load(1),
    ...             loader.load(2),
    ...             loader.load_many([2, 3]),
    ...         )
    ...         return results
    >>>
    >>> asyncio.run(example())
    ['user_1', 'user_2', ['user_2', 'user_3']]
    >>> _batches # The load function was called only once with deduplicated keys
    [[1, 2, 3]]

    Caching example:

    >>> _batches = list[Sequence[int]]()
    >>> async def caching_example():
    ...     async with DataLoader(load_users) as loader:
    ...         user_1 = await loader.load(1)
    ...         user_1_again = await loader.load(1)
    ...         return [user_1, user_1_again]
    >>>
    >>> asyncio.run(caching_example())
    ['user_1', 'user_1']
    >>> _batches
    [[1]]

    How it works
    ------------

    When you call load() or load_many(), the DataLoader doesn't immediately execute the
    load function. Instead, it schedules a task for the next event loop iteration,
    collects keys, and returns a Future immediately. If other load calls are made while
    the task is pending, their keys are collected and batched together. This works best
    when load calls are made in parallel through asyncio.gather() or tasks.
    """

    # Invariants maintained between calls and await points:
    # - If _keys_to_load is non-empty, then _scheduled_load_task exists
    # - _pending_results contains at least all keys from _keys_to_load
    # - Running load tasks never have overlapping keys

    load_fn: LoadFunction[_KeyType, _ResultType]
    max_batch_size: int | None
    cache: bool

    def __init__(
        self,
        load_fn: LoadFunction[_KeyType, _ResultType],
        max_batch_size: int | None = None,
        cache: bool = True,
        loop: asyncio.AbstractEventLoop | None = None,
        cache_map: MutableMapping[_KeyType, _ResultType] | None = None,
    ) -> None:
        """
        Initialize a DataLoader instance.

        :param load_fn: Async function that accepts a sequence of keys and returns results
            in the same order. Results can be values or Exception instances.
        :param max_batch_size: Maximum number of keys per batch. Default is None (unlimited).
        :param cache: Whether to cache successful results. Default is True.
        :param loop: Event loop to use. Only needed when calling load() outside an async context.
        :param cache_map: Custom cache storage to use. If not provided, a plain dict will be used.
        """
        self.load_fn = load_fn
        self._load_fn_name = f"{load_fn.__qualname__}" if hasattr(load_fn, "__qualname__") else f"{load_fn!r}"
        self.max_batch_size = max_batch_size
        self.cache = cache
        self.cache_map: MutableMapping[_KeyType, _ResultType] = cache_map if cache_map is not None else {}
        self._maybe_loop = loop

        self._keys_to_load: list[_KeyType] = []
        self._pending_results: dict[_KeyType, asyncio.Future[_ResultType]] = {}

        self._scheduled_load_task: asyncio.Task[None] | None = None
        self._running_load_tasks: set[asyncio.Task[None]] = set()
        self._task_counter: int = 0

        self._entered: bool = False

    def load(self, key: _KeyType) -> asyncio.Future[_ResultType]:
        if key in self.cache_map:
            future = self._loop.create_future()
            future.set_result(self.cache_map[key])
            return future

        future = self._pending_results.get(key)
        if future is not None:
            return future

        self._keys_to_load.append(key)
        self._pending_results[key] = future = self._loop.create_future()
        self._ensure_load_task_is_scheduled()

        return future

    def load_many(self, keys: Iterable[_KeyType]) -> asyncio.Future[list[_ResultType]]:
        return asyncio.gather(*(self.load(key) for key in keys))

    def clear(self, key: _KeyType) -> None:
        self.cache_map.pop(key, None)

    def clear_many(self, keys: Iterable[_KeyType]) -> None:
        for key in keys:
            self.cache_map.pop(key, None)

    def clear_all(self) -> None:
        self.cache_map.clear()

    def prime(self, key: _KeyType, value: _ResultType) -> None:
        self.cache_map[key] = value

    def prime_many(self, data: Mapping[_KeyType, _ResultType]) -> None:
        for key, value in data.items():
            self.cache_map[key] = value

    async def shutdown(self) -> ExceptionGroup | None:
        cancelled_tasks: list[asyncio.Task[None]] = []

        if self._scheduled_load_task is not None and not self._scheduled_load_task.done():
            self._scheduled_load_task.cancel()
            cancelled_tasks.append(self._scheduled_load_task)
            self._scheduled_load_task = None

        for task in self._running_load_tasks:
            if not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        self._running_load_tasks.clear()

        for future in self._pending_results.values():
            if not future.done():
                future.cancel()
        self._pending_results.clear()
        self._keys_to_load.clear()

        exceptions: list[Exception] = []
        for task in cancelled_tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception as e:
                exceptions.append(e)

        if exceptions:
            return ExceptionGroup("DataLoader shutdown encountered exceptions", exceptions)

    def _ensure_load_task_is_scheduled(self) -> None:
        if self._scheduled_load_task is not None:
            return

        self._task_counter += 1
        self._scheduled_load_task = self._loop.create_task(
            self._load_collected_keys(),
            name=f"DataLoader({self._load_fn_name})-{self._task_counter}",
        )

    async def _load_collected_keys(self) -> None:
        # Since we're here, the task is no longer pending, it's running
        assert self._scheduled_load_task is not None
        current_task = self._scheduled_load_task
        self._scheduled_load_task = None
        self._running_load_tasks.add(current_task)

        keys = self._deque_next_keys_batch()
        if len(self._keys_to_load) > 0:
            self._ensure_load_task_is_scheduled()

        try:
            results = await self.load_fn(keys)

            if len(results) != len(keys):
                raise ValueError("Wrong number of results returned by load_fn in DataLoader")

            for key, result in zip(keys, results, strict=True):
                self._fulfil_result(key, result)

        except (asyncio.CancelledError, Exception) as e:
            for key in keys:
                self._fulfil_result(key, e)
            return

        finally:
            self._running_load_tasks.discard(current_task)

    def _deque_next_keys_batch(self) -> list[_KeyType]:
        if self.max_batch_size is None or len(self._keys_to_load) <= self.max_batch_size:
            # We can avoid copying by swapping out _keys_to_load
            batch, keys_left = self._keys_to_load, []

        else:
            batch, keys_left = (
                self._keys_to_load[: self.max_batch_size],
                self._keys_to_load[self.max_batch_size :],
            )

        self._keys_to_load = keys_left
        return batch

    def _fulfil_result(
        self,
        key: _KeyType,
        result: _ResultType | Exception | asyncio.CancelledError,
    ) -> None:
        future = self._pending_results.pop(key, None)
        if future is None or future.done():
            return

        match result:
            case asyncio.CancelledError():
                future.cancel()
            case Exception() as exception:
                future.set_exception(exception)
            case _:
                future.set_result(result)
                if self.cache:
                    self.cache_map[key] = result

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("DataLoader instance already entered")
        self._entered = True

        if self._maybe_loop is None:
            # We're in async context so we can opportunistically get the running loop
            self._maybe_loop = asyncio.get_running_loop()

        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        if not self._entered:
            raise RuntimeError("DataLoader instance has not been entered")

        self._entered = False
        await self.shutdown()

    @property
    def _loop(self) -> asyncio.AbstractEventLoop:
        if self._maybe_loop is None:
            self._maybe_loop = asyncio.get_running_loop()

        return self._maybe_loop
