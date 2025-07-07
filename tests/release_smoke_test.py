"""
Small standalone test of dloader basics

This module will be run in a bare virtual environment with only the dloader package installed.
Don't try to import any other packages, in particular don't import pytest.
"""

import asyncio
import sys
import traceback
from collections.abc import Sequence


def test_can_import_dloader() -> None:
    """Test that we can import dloader."""
    try:
        import dloader  # type: ignore  # noqa: F401
    except ImportError as e:
        raise AssertionError("Could not import dloader package") from e

    try:
        from dloader import DataLoader  # type: ignore # noqa: F401
    except ImportError as e:
        raise AssertionError("Could not import Dataloader class from dloader package") from e


def test_run_simple_load() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        await asyncio.sleep(0.1)
        return [f"ok-{key}" for key in keys]

    async def run_load() -> None:
        from dloader import DataLoader

        loader = DataLoader(load_fn)

        future_1 = loader.load(1)
        future_2 = loader.load(2)

        results = await asyncio.gather(future_1, future_2)
        assert results == ["ok-1", "ok-2"]

    try:
        asyncio.run(run_load(), debug=True)
    except Exception as e:
        raise AssertionError("Failed to run a simple load with Dataloader") from e


if __name__ == "__main__":
    try:
        test_can_import_dloader()
        print("Package has been successfully imported")
        test_run_simple_load()
        print("Simple load has been successfully executed")

    except AssertionError:
        print("Smoke test failed")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
