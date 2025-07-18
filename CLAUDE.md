# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

dloader is a Python package that provides an asynchronous DataLoader implementation for Python, providing batching and
caching of data loads. It follows the DataLoader pattern commonly used in GraphQL servers.

## Development Commands

All commands use the `just` task runner (@Justfile):

```bash
just lint         # Run linting checks (ruff format & check)
just fix          # Auto-fix linting issues
just test         # Run tests with pytest
just typecheck    # Run type checking with pyright
just qa           # Run full quality assurance (fix, lint, typecheck, test)
just clean        # Clean cache and compiled files
just deps-upgrade # Upgrade dependencies
```

To run a single test:

```bash
uv run pytest tests/test_dataloader.py::test_name -v
```

## Architecture

The library consists of a single main module with a generic `DataLoader[K, V]` class that:

1. **Batches requests**: Multiple `load(key)` calls are automatically batched into a single `load_fn(keys)` call
2. **Caches results**: Results are cached and reused for subsequent requests with the same key
3. **Handles errors**: Exceptions are properly propagated per key
4. **Deduplicates**: Same keys within a batch are automatically deduplicated
5. **Manages concurrency**: Handles pending and running load tasks efficiently

Key points:

- Fully typed, including using generics where appropriate
- Careful management of asyncio tasks to prevent resource leakage
- Small, efficient and no external dependencies
- Build for asyncio, ignores normal thread-safety

## Code Quality Standards

- **Type checking**: Pyright in strict mode - all code must pass strict type checking
- **Python version**: As library, supports Python versions 3.11 and later
- **Style**: Ruff for formatting and linting
- **Testing**: All new features require tests using pytest with pytest-asyncio
- **Commenting**: Only add comments for things that are unexpected, surprising, uncommon. Never add comments that just
  restate code

## Testing Guidelines

- Tests are located in `@/tests/` and cover all features
- When adding new features, ensure tests cover both success and error cases, especially for concurrent scenarios
- Test functions names should explain what behaviour is being tested without being too long
- Avoid checking class internals in tests, always try to test via class public interface or test helpers

## Naming Conventions

- In variable names with numbers use underscore to separate words and numbers, like `variable_1` and `load_1_result`

## Documentation

- Docstrings need to be concise and state only things that are not obvious from the symbol name or signature
- No bullet points in docs unless explicitly asked

## GitHub Actions style

- Ensure that each step has a human readable name that's an imperative sentence, like "Install dependencies" or "Run
  tests"
- Ensure that actions have consistent style (including whitespace), properties are in the same order, etc.

## Commit Guidelines

- When committing, commit description should contain the reason for the change, avoid just describing the change
- Don't mention Claude and don't insert Claude as coauthor
