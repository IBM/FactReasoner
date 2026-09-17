# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Persistent background event loop for dispatching coroutines from sync code.

Keeps the loop open, so httpx cleanup tasks run to completion at whatever
depth and pace they need.  The same loop ``id`` is reused across calls, so
Mellea's ``OpenAIBackend._async_client`` cache returns the same
``AsyncOpenAI``/httpx ``AsyncClient`` instance instead of creating a new one
(and abandoned one) per call.

Thread-safety
-------------
``_get_loop()`` uses a double-checked lock so only one loop is ever created.
The daemon thread is never joined — it exits automatically when the process ends.

Usage
-----
    from fact_reasoner.core._async_runner import run_coroutine

    result = run_coroutine(some_async_fn(...))
"""

import asyncio
import threading

_loop: asyncio.AbstractEventLoop | None = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the shared background event loop, creating it on first call."""
    global _loop
    # Fast path: already running.
    if _loop is not None and not _loop.is_closed():
        return _loop
    with _loop_lock:
        # Re-check under the lock.
        if _loop is not None and not _loop.is_closed():
            return _loop
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True, name="async-runner").start()
        _loop = loop
        return loop


def run_coroutine(coro) -> object:
    """
    Run *coro* on the shared background event loop and block until it completes.

    Safe to call from:
    - plain synchronous code (no running loop)
    - inside a ``ThreadPoolExecutor`` worker thread
    - inside a request handler served by an async web server (running loop on another thread)

    The coroutine — and all tasks it schedules (e.g. httpx cleanup) — run on the
    same persistent loop, so they never encounter a closed loop.
    """
    loop = _get_loop()
    # Make the persistent loop the "current" loop for this thread so that anyio's
    # asyncio backend picks it up via asyncio.get_event_loop() when binding transports.
    asyncio.set_event_loop(loop)
    return asyncio.run_coroutine_threadsafe(coro, loop).result()
