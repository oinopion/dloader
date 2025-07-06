import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from dloader import LoadFunction


class InstrumentedLoad(LoadFunction[int, str]):
    """Test helper that records batch calls and controls execution timing."""

    batches: list[list[int]]
    """History of all batch calls made to this load function."""
    proceed_signal: asyncio.Event
    """When unset, load function will block until set."""
    load_started: asyncio.Event
    """Set when load function begins processing, useful for synchronization."""
    side_effect: Callable[[list[int]], Any] | None = None
    """Optional function to call with keys, can raise exceptions for error testing."""

    def __init__(
        self,
        db: Mapping[int, str | Exception] | None = None,
        proceed_immediately: bool = True,
        side_effect: Callable[[list[int]], Any] | None = None,
    ) -> None:
        """
        :param db: Mock database mapping keys to values or exceptions. Defaults to {1: "data-1", 2: "data-2", ...}
        :param proceed_signal: If False, load function will block until proceed_signal.set() is called
        :param side_effect: Optional function called with keys; if it returns an exception, that exception is raised
        """
        self.db = db if db is not None else {i: f"data-{i}" for i in range(1, 10)}
        self.side_effect = side_effect

        self.batches: list[list[int]] = []
        self.proceed_signal = asyncio.Event()
        if proceed_immediately:
            self.proceed_signal.set()
        self.load_started = asyncio.Event()

    async def __call__(self, keys: Sequence[int]) -> Sequence[str | Exception]:
        self.batches.append(list(keys))
        self.load_started.set()
        await self.proceed_signal.wait()
        if self.side_effect is not None:
            result = self.side_effect(list(keys))
            if isinstance(result, Exception):
                raise result

        return [self.db[key] for key in keys]
