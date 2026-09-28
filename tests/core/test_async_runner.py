# coding=utf-8
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

"""Tests for the persistent background event loop runner."""

import asyncio
import warnings

import pytest

from fact_reasoner.core._async_runner import run_coroutine


async def _loop_id() -> int:
    return id(asyncio.get_running_loop())


def test_reuses_one_loop_across_calls():
    # Mellea caches its async client by id(loop), so every call must land on
    # the same live loop rather than a fresh (and later closed) one.
    ids = {run_coroutine(_loop_id()) for _ in range(20)}
    assert len(ids) == 1


def test_callable_from_inside_a_running_loop():
    async def main():
        return run_coroutine(_loop_id()), id(asyncio.get_running_loop())

    runner_id, outer_id = asyncio.run(main())
    assert runner_id != outer_id


def test_does_not_hijack_caller_event_loop():
    run_coroutine(asyncio.sleep(0))
    # The caller's thread must not be left pointing at the runner's loop,
    # which is already running on another thread.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            current = asyncio.get_event_loop()
        except RuntimeError:
            return  # no current loop at all -- nothing was hijacked
    try:
        assert not current.is_running()
    finally:
        current.close()
        asyncio.set_event_loop(None)


def test_raises_instead_of_deadlocking_on_runner_loop():
    async def reenter():
        coro = asyncio.sleep(0)
        try:
            run_coroutine(coro)
        finally:
            coro.close()

    with pytest.raises(RuntimeError, match="async-runner loop"):
        run_coroutine(reenter())
