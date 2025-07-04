from __future__ import annotations

import asyncio
from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Generic, Protocol, TypeVar

__all__ = (
    "DataLoader",
    "LoadFunction",
)

_K = TypeVar("_K", bound=Hashable, contravariant=True)
_V = TypeVar("_V", covariant=True)


class LoadFunction(Protocol, Generic[_K, _V]):
    async def __call__(
        self,
        keys: Sequence[_K],
    ) -> Sequence[_V | Exception]: ...


_KeyType = TypeVar("_KeyType", bound=Hashable)
_ResultType = TypeVar("_ResultType")


class DataLoader(Generic[_KeyType, _ResultType]):
    load_fn: LoadFunction[_KeyType, _ResultType]
    max_batch_size: int | None
    cache: bool

    def __init__(
        self,
        load_fn: LoadFunction[_KeyType, _ResultType],
        max_batch_size: int | None = None,
        cache: bool = True,
    ) -> None:
        self.load_fn = load_fn
        self.max_batch_size = max_batch_size
        self.cache = cache

        # Cache storage for completed results
        self._cache_store = dict[_KeyType, _ResultType]()

        # Keys to load are collected temporarily until the a load task starts
        self._keys_to_load = list[_KeyType]()
        self._pending_results = dict[_KeyType, asyncio.Future[_ResultType]]()
        # Pending means waiting for the loop to pick it up, we keep collecting keys until it does
        self._scheduled_load_task: asyncio.Task[None] | None = None
        # Running means the loop has picked it up and is currently executing it; there can be
        # multiple running tasks, each with its own set of keys
        self._running_load_tasks = set[asyncio.Task[None]]()

    def load(self, key: _KeyType) -> asyncio.Future[_ResultType]:
        loop = asyncio.get_event_loop()

        if self.cache and key in self._cache_store:
            future = loop.create_future()
            future.set_result(self._cache_store[key])
            return future

        future = self._pending_results.get(key)
        if future is not None:
            return future

        self._keys_to_load.append(key)
        self._pending_results[key] = future = loop.create_future()
        self._ensure_load_task_is_scheduled()

        return future

    def load_many(self, keys: Iterable[_KeyType]) -> asyncio.Future[list[_ResultType]]:
        return asyncio.gather(*(self.load(key) for key in keys))

    def clear(self, key: _KeyType) -> None:
        self._cache_store.pop(key, None)

    def clear_many(self, keys: Iterable[_KeyType]) -> None:
        for key in keys:
            self._cache_store.pop(key, None)

    def clear_all(self) -> None:
        self._cache_store.clear()

    def prime(self, key: _KeyType, value: _ResultType, force: bool = False) -> None: ...

    def prime_many(self, data: Mapping[_KeyType, _ResultType], force: bool = False) -> None: ...

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

        exceptions: list[Exception] = []
        for task in cancelled_tasks:
            try:
                await task
            except asyncio.CancelledError:
                continue
            except Exception as e:
                exceptions.append(e)

        if exceptions:
            return ExceptionGroup(
                "DataLoader shutdown encountered exceptions",
                exceptions,
            )

    def _ensure_load_task_is_scheduled(self) -> None:
        if self._scheduled_load_task is not None:
            return

        loop = asyncio.get_event_loop()
        coroutine = self._load_collected_keys()
        task_name = f"DataLoader({self.load_fn.__qualname__})._load_gathered_keys"
        self._scheduled_load_task = loop.create_task(coroutine, name=task_name)

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
            assert current_task is not None
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
                    self._cache_store[key] = result
