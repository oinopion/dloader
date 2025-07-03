from __future__ import annotations

import asyncio


import itertools
from typing import (
    Sequence,
    TypeVar,
    Generic,
    Hashable,
    Iterable,
    Mapping,
)


KeyType = TypeVar("KeyType", bound=Hashable)
ResultType = TypeVar("ResultType")


class LoadFunction(Generic[KeyType, ResultType]):
    async def __call__(
        self,
        keys: Sequence[KeyType],
    ) -> Sequence[ResultType | Exception]: ...


class DataLoader(Generic[KeyType, ResultType]):
    load_fn: LoadFunction[KeyType, ResultType]
    max_batch_size: int | None
    cache: bool

    def __init__(
        self,
        load_fn: LoadFunction[KeyType, ResultType],
        max_batch_size: int | None = None,
        cache: bool = True,
    ) -> None:
        self.load_fn = load_fn
        self.max_batch_size = max_batch_size
        self.cache = cache

        # Keys to load are collected temporarily until the a load task starts
        self._keys_to_load = dict[KeyType, asyncio.Future[ResultType]]()
        # Pending means waiting for the loop to pick it up, we keep collecting keys until it does
        self._pending_load_task: asyncio.Task[None] | None = None
        # Running means the loop has picked it up and is currently executing it; there can be
        # multiple running tasks, each with its own set of keys
        self._running_load_tasks = set[asyncio.Task[None]]()
        # Cache storage for completed results
        self._cache = dict[KeyType, ResultType]()

    def load(self, key: KeyType) -> asyncio.Future[ResultType]:
        loop = asyncio.get_event_loop()

        # Check cache first if caching is enabled
        if self.cache and key in self._cache:
            future = loop.create_future()
            future.set_result(self._cache[key])
            return future

        future = self._keys_to_load.get(key)
        if future is not None:
            return future

        self._keys_to_load[key] = future = loop.create_future()
        self._ensure_load_task_is_pending()

        return future

    def load_many(self, keys: Iterable[KeyType]) -> asyncio.Future[list[ResultType]]:
        return asyncio.gather(*[self.load(key) for key in keys])

    def clear(self, key: KeyType) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear_many(self, keys: Iterable[KeyType]) -> None:
        for key in keys:
            self.clear(key)

    def clear_all(self) -> None:
        self._cache.clear()

    def prime(self, key: KeyType, value: ResultType, force: bool = False) -> None: ...

    def prime_many(
        self, data: Mapping[KeyType, ResultType], force: bool = False
    ) -> None: ...

    async def shutdown(self) -> ExceptionGroup | None:
        cancelled_tasks = []

        if self._pending_load_task is not None and not self._pending_load_task.done():
            self._pending_load_task.cancel()
            cancelled_tasks.append(self._pending_load_task)
            self._pending_load_task = None

        for task in self._running_load_tasks:
            if not task.done():
                task.cancel()
                cancelled_tasks.append(task)
        self._running_load_tasks.clear()

        exceptions = []
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

    def _ensure_load_task_is_pending(self) -> None:
        if self._pending_load_task is not None:
            return

        loop = asyncio.get_event_loop()
        coroutine = self._load_collected_keys()
        task_name = f"DataLoader({self.load_fn.__qualname__})._load_gathered_keys"
        self._pending_load_task = loop.create_task(coroutine, name=task_name)

    async def _load_collected_keys(self) -> None:
        # Since we're here, the task is no longer pending, it's running
        current_task = self._pending_load_task
        self._pending_load_task = None
        self._running_load_tasks.add(current_task)

        batch = self._collect_one_batch()
        if len(self._keys_to_load) > 0:
            self._ensure_load_task_is_pending()

        batch_keys = list(batch.keys())
        try:
            results = await self.load_fn(batch_keys)

            if len(results) != len(batch_keys):
                raise ValueError("Wrong number of results returned by load_fn")

            for (key, future), result in zip(batch.items(), results):
                if future.done():
                    continue

                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
                    # Store successful results in cache if caching is enabled
                    if self.cache:
                        self._cache[key] = result

        except asyncio.CancelledError:
            for future in batch.values():
                if not future.done():
                    future.cancel()
            return

        except Exception as e:
            for future in batch.values():
                if not future.done():
                    future.set_exception(e)
            return

        finally:
            self._running_load_tasks.discard(current_task)

    def _collect_one_batch(self) -> dict[KeyType, asyncio.Future[ResultType]]:
        if (
            self.max_batch_size is None
            or len(self._keys_to_load) <= self.max_batch_size
        ):
            # We can avoid copying by swapping out _keys_to_load
            batch = self._keys_to_load
            self._keys_to_load = {}
            return batch

        batch_keys = list(
            itertools.islice(self._keys_to_load.keys(), self.max_batch_size)
        )
        batch = {}
        for key in batch_keys:
            batch[key] = self._keys_to_load.pop(key)
        return batch
