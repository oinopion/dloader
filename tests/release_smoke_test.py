"""
Small standalone test of dloader basics

This module will be run in a bare virtual environment with only the dloader package installed.
Don't try to import any other packages, in particular don't import pytest.
"""

import asyncio
import importlib.metadata
import sys
import traceback
from collections.abc import Sequence


def test_can_import_dloader() -> None:
    """Test that we can import dloader."""
    try:
        import dloader  # type: ignore
    except ImportError as e:
        raise AssertionError("Could not import dloader package") from e

    try:
        from dloader import DataLoader  # type: ignore # noqa: F401
    except ImportError as e:
        raise AssertionError("Could not import DataLoader class from dloader package") from e

    if not hasattr(dloader, "__version__"):
        raise AssertionError("dloader package does not have __version__ attribute")

    if not isinstance(dloader.__version__, str):  # type: ignore
        raise AssertionError(f"dloader.__version__ is not a string: {type(dloader.__version__)}")

    if not dloader.__version__:
        raise AssertionError("dloader.__version__ is empty")

    installed_version = importlib.metadata.version("dloader")
    if installed_version != dloader.__version__:
        raise AssertionError(
            f"Version mismatch: installed version is {installed_version!r}, but __version__ is {dloader.__version__!r}"
        )

    print(f"Package has been successfully imported with version {dloader.__version__}")


def test_run_simple_load() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        await asyncio.sleep(0.1)
        return [f"ok-{key}" for key in keys]

    async def run_load() -> None:
        from dloader.dataloader import DataLoader

        loader = DataLoader(load_fn)

        future_1 = loader.load(1)
        future_2 = loader.load(2)

        results = await asyncio.gather(future_1, future_2)
        assert results == ["ok-1", "ok-2"]

    try:
        asyncio.run(run_load(), debug=True)
    except Exception as e:
        raise AssertionError("Failed to run a simple load with DataLoader") from e

    print("Simple load has been successfully executed")


if __name__ == "__main__":
    try:
        test_can_import_dloader()
        test_run_simple_load()

    except AssertionError:
        print("Smoke test failed")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
