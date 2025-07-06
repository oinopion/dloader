# dloader

A Python implementation of the DataLoader pattern for data fetching with automatic batching.

## Installation

```bash
uv add dloader
# or
pip install dloader
```

## Quick Start

```python
import asyncio
from dloader import DataLoader

# Define a batch loading function
async def batch_load_users(user_ids):
    # Fetch multiple users at once (e.g., from a database)
    users = await db_client.get_many(User, user_ids)
    return users

# Create a DataLoader instance
async with DataLoader(batch_load_users) as user_loader:
    # These calls are automatically batched
    user_1, user_2, other_users = await asyncio.gather(
        user_loader.load(1),
        user_loader.load(2),
        user_loader.load_many([3, 4, 5]),
    )
```

## How It Works

When you call `load()` or `load_many()`, the DataLoader doesn't immediately execute the load function. Instead, it schedules a task for the next event loop iteration, collects keys, and returns a Future immediately. If other load calls are made while the task is pending, their keys are collected and batched together. This works best when load calls are made in parallel through asyncio.gather() or tasks.

## API Reference

### DataLoader

```python
DataLoader[K, V](
    load_fn: LoadFunction[K, V],
    max_batch_size: int | None = None,
    cache: bool = True,
    loop: asyncio.AbstractEventLoop | None = None,
    cache_map: MutableMapping[K, V] | None = None,
)
```

#### Parameters

- `load_fn`: Async callable that accepts a sequence of keys and returns results in the same order. Results can be values or Exception instances.
- `max_batch_size`: Maximum number of keys per batch. Default is None (unlimited).
- `cache`: Whether to cache successful results. Default is True.
- `loop`: Event loop to use. Only needed when calling load() outside an async context.
- `cache_map`: Custom cache storage to use. If not provided, a plain dict will be used.

#### Methods

##### `load(key: K) -> Future[V]`

Load a single value by its key. Returns a Future that resolves to the value.

##### `load_many(keys: Iterable[K]) -> Future[list[V]]`

Load multiple values. Returns a Future that resolves to a list of values.

##### `clear(key: K) -> None`

Remove a single key from the cache.

##### `clear_many(keys: Iterable[K]) -> None`

Remove multiple keys from the cache.

##### `clear_all() -> None`

Clear the entire cache.

##### `prime(key: K, value: V) -> None`

Pre-populate the cache with a key-value pair.

##### `prime_many(data: Mapping[K, V]) -> None`

Pre-populate the cache with multiple key-value pairs.

##### `shutdown() -> ExceptionGroup | None`

Clean up all pending operations. Called automatically when using as a context manager.

### LoadFunction Protocol

The load function must follow this protocol:

```python
class LoadFunction(Protocol, Generic[K, V]):
    async def __call__(self, keys: Sequence[K], /) -> Sequence[V | Exception]: ...

# This is equivalent to:
async def load_fn(keys: Sequence[K]) -> Sequence[V | Exception]:
    ...
```

This callable will receive a sequence of keys and has to return a sequence of results. The sequence of results has to be
the same length as the keys and results have to be in the same order as keys, i.e., a result in the position X corresponds
to a key in the position X. A result can also be an instance of `Exception` in which case it will be set as an exception
on the respective future, which will raise that exception when the future is awaited on.

## Best Practices

- Use as context manager to guarantee all tasks and futures are correctly cleaned up. Alternatively, make sure to call
  `shutdown()` when you're done with the dataloader.
- Use `asyncio.gather()` or other asyncio concurrency patterns, like `strawberry`'s async GraphQL resolvers.

## License

MIT License - see LICENSE file for details.
