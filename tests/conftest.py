from __future__ import annotations

import asyncio

import pytest

EVENT_LOOP_POLICIES: dict[str, asyncio.AbstractEventLoopPolicy] = {
    "default": asyncio.DefaultEventLoopPolicy(),
}

try:
    import uvloop

    EVENT_LOOP_POLICIES["uvloop"] = uvloop.EventLoopPolicy()
except ImportError:
    pass


@pytest.fixture(params=EVENT_LOOP_POLICIES.items(), ids=lambda p: p[0])
def event_loop_policy(request: pytest.FixtureRequest) -> asyncio.AbstractEventLoopPolicy:
    return request.param[1]
