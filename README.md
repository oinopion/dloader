# dloader

A Python implementation of the DataLoader pattern for data fetching with automatic batching.

## Features

- Automatically batches and deduplicates concurrent loads
- Caches results and allows custom in-memory caches
- Manages asyncio tasks and allows proper cleanup
- Small, focused and free of other dependencies
- Fully typed and fully tested

## Installation

```bash
uv add dloader
# or
pip install dloader
```

## Documentation

You can find documentation at: https://oinopion.github.io/dloader

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

## Versioning

This project follows CalVer. Expect couple of release per year with very minor changes.

## License

MIT License - see [LICENSE](./LICENSE) file for details.
