from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence

import pytest

from dloader import DataLoader

pytest.mark.asyncio(loop_scope="function")


async def test_basic_serial_loading_returns_loaded_results() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

    assert await loader.load(1) == "data-1"
    assert await loader.load(2) == "data-2"
    assert await loader.load_many([3, 4]) == ["data-3", "data-4"]


async def test_basic_batch_loading_returns_loaded_results() -> None:
    batches = list[Sequence[int]]()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        batches.append(keys)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)
    results = await asyncio.gather(
        loader.load(1),
        loader.load(2),
        loader.load_many([3, 4]),
        return_exceptions=True,
    )

    assert results == ["data-1", "data-2", ["data-3", "data-4"]]
    assert len(batches) == 1
    assert batches[0] == [1, 2, 3, 4]


async def test_returned_exceptions_are_set_as_future_exceptions() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str | Exception]:
        return [f"data-{key}" if key % 2 != 0 else ValueError(f"Error loading key {key}") for key in keys]

    loader = DataLoader(load_fn=load_fn)

    assert await loader.load(1) == "data-1"
    with pytest.raises(ValueError, match="Error loading key 2"):
        await loader.load(2)
    assert await loader.load(3) == "data-3"


async def test_exception_from_load_fn_is_set_as_future_exception() -> None:
    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        raise RuntimeError(f"Failed load: {', '.join(map(str, keys))}")

    loader = DataLoader(load_fn=load_fn)

    with pytest.raises(RuntimeError, match=r"Failed load: 1"):
        await loader.load(1)

    with pytest.raises(RuntimeError, match=r"Failed load: 2, 3"):
        await loader.load_many([2, 3])


async def test_shutting_down_cancels_all_pending_tasks() -> None:
    load_in_progress = asyncio.Event()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        load_in_progress.set()
        await asyncio.sleep(10)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

    async def cancellable_action() -> str:
        result = await loader.load(1)
        return f"Loaded: {result}"

    task = asyncio.create_task(cancellable_action())
    await load_in_progress.wait()

    await loader.shutdown()

    assert task.cancelled()

    with pytest.raises(asyncio.CancelledError):
        await task  # Ensure the task raises CancelledError


async def test_dataloader_honors_max_batch_size() -> None:
    batches = list[Sequence[int]]()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        batches.append(keys)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn, max_batch_size=3)

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

    # Should have been split into 3 batches: [1,2,3], [4,5,6], [7]
    assert len(batches) == 3
    assert batches[0] == [1, 2, 3]
    assert batches[1] == [4, 5, 6]
    assert batches[2] == [7]


async def test_exception_when_load_fn_returns_wrong_number_of_results() -> None:
    async def load_fn_too_few(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys[:-1]]

    async def load_fn_too_many(keys: Sequence[int]) -> Sequence[str]:
        return [f"data-{key}" for key in keys] + ["extra"]

    loader_few = DataLoader(load_fn=load_fn_too_few)

    with pytest.raises(ValueError, match="Wrong number of results returned by load_fn"):
        await asyncio.gather(
            loader_few.load(1),
            loader_few.load(2),
            loader_few.load(3),
        )

    loader_many = DataLoader(load_fn=load_fn_too_many)

    with pytest.raises(ValueError, match="Wrong number of results returned by load_fn"):
        await asyncio.gather(
            loader_many.load(4),
            loader_many.load(5),
            loader_many.load(6),
        )


async def test_loads_are_minimised_under_overlapping_requests() -> None:
    batches = list[Sequence[int]]()
    load_started = asyncio.Event()
    green_light = asyncio.Event()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        batches.append(keys)
        load_started.set()
        await green_light.wait()
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

    future_1 = loader.load(1)
    future_2 = loader.load(2)
    await load_started.wait()

    future_3 = loader.load(2)
    future_4 = loader.load(3)

    green_light.set()
    results = await asyncio.gather(future_1, future_2, future_3, future_4)

    assert results == ["data-1", "data-2", "data-2", "data-3"]
    assert batches == [
        [1, 2],
        [3],
    ]


async def test_caching_with_concurrent_loads() -> None:
    load_counter = Counter[int]()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        load_counter.update(keys)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

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

    assert load_counter[1] == 1
    assert load_counter[2] == 1
    assert load_counter[3] == 1
    assert load_counter[4] == 1
    assert load_counter[5] == 1
    assert load_counter[6] == 1


async def test_prime_single_key() -> None:
    batches = list[Sequence[int]]()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        batches.append(keys)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

    loader.prime(1, "primed-data-1")

    result = await loader.load(1)
    assert result == "primed-data-1"
    assert len(batches) == 0

    result_2 = await loader.load(2)
    assert result_2 == "data-2"
    assert len(batches) == 1
    assert batches[0] == [2]

    loader.prime(2, "new-primed-data-2")
    result_3 = await loader.load(2)
    assert result_3 == "new-primed-data-2"
    assert len(batches) == 1


async def test_prime_many_keys() -> None:
    batches = list[Sequence[int]]()

    async def load_fn(keys: Sequence[int]) -> Sequence[str]:
        batches.append(keys)
        return [f"data-{key}" for key in keys]

    loader = DataLoader(load_fn=load_fn)

    loader.prime_many(
        {
            1: "primed-1",
            2: "primed-2",
            3: "primed-3",
        }
    )

    results = await loader.load_many([1, 2, 3])
    assert results == ["primed-1", "primed-2", "primed-3"]
    assert len(batches) == 0

    results_2 = await loader.load_many([2, 4, 5])
    assert results_2 == ["primed-2", "data-4", "data-5"]
    assert len(batches) == 1
    assert batches[0] == [4, 5]

    loader.prime_many(
        {
            4: "new-primed-4",
            5: "new-primed-5",
            6: "primed-6",
        }
    )

    results_3 = await loader.load_many([4, 5, 6])
    assert results_3 == ["new-primed-4", "new-primed-5", "primed-6"]
    assert len(batches) == 1
