from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from cachetools import Cache
from testing_tools import InstrumentedLoad

from dloader import DataLoader


async def test_basic_serial_loading_returns_loaded_results() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys]

    async with DataLoader(load_fn=load_fn) as loader:
        assert await loader.load(1) == "data-1"
        assert await loader.load(2) == "data-2"
        assert await loader.load_many([3, 4]) == ["data-3", "data-4"]


async def test_basic_batch_loading_returns_loaded_results() -> None:
    load_fn = InstrumentedLoad()
    async with DataLoader(load_fn=load_fn) as loader:
        results = await asyncio.gather(
            loader.load(1),
            loader.load(2),
            loader.load_many([3, 4]),
            return_exceptions=True,
        )

        assert results == ["data-1", "data-2", ["data-3", "data-4"]]
        assert load_fn.batches == [
            [1, 2, 3, 4],
        ]


async def test_returned_exceptions_are_set_as_future_exceptions() -> None:
    load_fn = InstrumentedLoad({1: "data-1", 2: ValueError("Error loading key 2"), 3: "data-3"})
    async with DataLoader(load_fn=load_fn) as loader:
        assert await loader.load(1) == "data-1"
        with pytest.raises(ValueError, match="Error loading key 2"):
            await loader.load(2)
        assert await loader.load(3) == "data-3"


async def test_exception_from_load_fn_is_set_as_future_exception() -> None:
    load_fn = InstrumentedLoad(side_effect=lambda k: RuntimeError(f"Failed load: {k}"))

    async with DataLoader(load_fn=load_fn) as loader:
        with pytest.raises(RuntimeError, match=r"Failed load: \[1\]"):
            await loader.load(1)

        with pytest.raises(RuntimeError, match=r"Failed load: \[2, 3\]"):
            await asyncio.gather(
                loader.load(2),
                loader.load(3),
            )


async def test_shutting_down_cancels_all_pending_tasks() -> None:
    load_fn = InstrumentedLoad(proceed_immediately=False)
    loader = DataLoader(load_fn=load_fn, loop=asyncio.get_event_loop())

    future = loader.load(1)
    await load_fn.load_started.wait()

    await loader.shutdown()

    assert future.cancelled()
    with pytest.raises(asyncio.CancelledError):
        await future


async def test_using_dataloader_as_context_manager_cancels_pending_tasks() -> None:
    load_fn = InstrumentedLoad(proceed_immediately=False)

    async with DataLoader(load_fn=load_fn) as loader:
        future = loader.load(1)
        await load_fn.load_started.wait()

    assert future.cancelled()

    with pytest.raises(asyncio.CancelledError):
        await future


async def test_dataloader_respects_max_batch_size() -> None:
    load_fn = InstrumentedLoad()
    async with DataLoader(load_fn=load_fn, max_batch_size=3) as loader:
        results = await asyncio.gather(
            loader.load(1),
            loader.load(2),
            loader.load(3),
            loader.load(4),
            loader.load_many([5, 6, 7]),
        )

        assert results == [
            "data-1",
            "data-2",
            "data-3",
            "data-4",
            ["data-5", "data-6", "data-7"],
        ]

        assert load_fn.batches == [
            [1, 2, 3],
            [4, 5, 6],
            [7],
        ]


async def test_dataloader_can_use_custom_cache_map() -> None:
    load_fn = InstrumentedLoad()

    cache_map = Cache[int, str](maxsize=2)
    async with DataLoader(load_fn=load_fn, cache=True, cache_map=cache_map) as loader:
        assert await loader.load_many([1, 2]) == ["data-1", "data-2"]
        assert list(cache_map.keys()) == [1, 2]
        assert await loader.load_many([2, 3]) == ["data-2", "data-3"]
        assert list(cache_map.keys()) == [2, 3]
        assert await loader.load_many([3, 1]) == ["data-3", "data-1"]
        assert list(cache_map.keys()) == [3, 1]

        assert load_fn.batches == [
            [1, 2],
            [3],
            [1],
        ]


async def test_exception_when_load_fn_returns_wrong_number_of_results() -> None:
    async def load_fn_too_few(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys[:-1]]

    async with DataLoader(load_fn=load_fn_too_few) as loader_few:
        with pytest.raises(ValueError, match="Wrong number of results returned by load_fn"):
            await asyncio.gather(
                loader_few.load(1),
                loader_few.load(2),
                loader_few.load(3),
            )

    async def load_fn_too_many(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys] + ["extra"]

    async with DataLoader(load_fn=load_fn_too_many) as loader_many:
        with pytest.raises(ValueError, match="Wrong number of results returned by load_fn"):
            await asyncio.gather(
                loader_many.load(4),
                loader_many.load(5),
                loader_many.load(6),
            )


async def test_loads_are_minimised_under_overlapping_requests() -> None:
    load_fn = InstrumentedLoad(proceed_immediately=False)

    async with DataLoader(load_fn=load_fn) as loader:
        future_1 = loader.load(1)
        future_2 = loader.load(2)
        await load_fn.load_started.wait()

        future_3 = loader.load(2)
        future_4 = loader.load(3)

        load_fn.proceed_signal.set()
        results = await asyncio.gather(future_1, future_2, future_3, future_4)

        assert results == ["data-1", "data-2", "data-2", "data-3"]
        assert load_fn.batches == [
            [1, 2],
            [3],
        ]


async def test_dataloader_caches_results() -> None:
    load_fn = InstrumentedLoad()

    async with DataLoader(load_fn=load_fn) as loader:
        results = await asyncio.gather(
            loader.load(1),
            loader.load(2),
            loader.load_many([3, 4]),
            loader.load(1),
            loader.load(2),
        )

        assert results == ["data-1", "data-2", ["data-3", "data-4"], "data-1", "data-2"]

        results_2 = await asyncio.gather(
            loader.load(1),
            loader.load(5),
            loader.load_many([2, 3, 6]),
            loader.load(5),
        )

        assert results_2 == ["data-1", "data-5", ["data-2", "data-3", "data-6"], "data-5"]

        assert load_fn.batches == [
            [1, 2, 3, 4],
            [5, 6],
        ]

        results_3 = await asyncio.gather(
            loader.load(1),
            loader.load(2),
            loader.load_many([3, 4]),
        )

        assert results_3 == ["data-1", "data-2", ["data-3", "data-4"]]
        assert len(load_fn.batches) == 2  # No new batches should be created


async def test_priming_adds_result_to_cache() -> None:
    load_fn = InstrumentedLoad()

    async with DataLoader(load_fn=load_fn) as loader:
        loader.prime(1, "primed-data-1")

        result = await loader.load(1)
        assert result == "primed-data-1"
        assert len(load_fn.batches) == 0

        result_2 = await loader.load(2)
        assert result_2 == "data-2"

        assert load_fn.batches == [[2]]

        loader.prime(2, "new-primed-data-2")
        result_3 = await loader.load(2)
        assert result_3 == "new-primed-data-2"
        assert len(load_fn.batches) == 1


async def test_prime_many_keys() -> None:
    load_fn = InstrumentedLoad()

    async with DataLoader(load_fn=load_fn) as loader:
        loader.prime_many(
            {
                1: "primed-1",
                2: "primed-2",
                3: "primed-3",
            }
        )

        results = await loader.load_many([1, 2, 3])
        assert results == ["primed-1", "primed-2", "primed-3"]
        assert len(load_fn.batches) == 0

        results_2 = await loader.load_many([2, 4, 5])
        assert results_2 == ["primed-2", "data-4", "data-5"]

        assert load_fn.batches == [[4, 5]]

        loader.prime_many(
            {
                4: "new-primed-4",
                5: "new-primed-5",
                6: "primed-6",
            }
        )

        results_3 = await loader.load_many([4, 5, 6])
        assert results_3 == ["new-primed-4", "new-primed-5", "primed-6"]
        assert len(load_fn.batches) == 1


async def test_shutdown_with_pending_keys_should_cancel_futures() -> None:
    load_fn = InstrumentedLoad()

    loader = DataLoader(load_fn)

    future_1 = loader.load(1)
    future_2 = loader.load(2)

    await loader.shutdown()

    with pytest.raises(asyncio.CancelledError):
        await future_1
    with pytest.raises(asyncio.CancelledError):
        await future_2

    assert len(load_fn.batches) == 0
