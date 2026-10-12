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

"""Unit tests for fact_reasoner.telemetry and context propagation."""

import asyncio

import pytest

from fact_reasoner import telemetry
from fact_reasoner.core._async_runner import run_coroutine

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.metrics import MeterProvider  # noqa: E402
from opentelemetry.sdk.metrics.export import InMemoryMetricReader  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)


@pytest.fixture
def spans(monkeypatch):
    """Route fact_reasoner spans to an in-memory exporter."""
    exporter = InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(telemetry, "_tracer", provider.get_tracer("test"))
    return exporter


@pytest.fixture
def metric_reader(monkeypatch):
    """Route fact_reasoner metrics to an in-memory reader."""
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    monkeypatch.setattr(
        telemetry, "_stage_duration", meter.create_histogram("stage.duration")
    )
    monkeypatch.setattr(telemetry, "_fallbacks", meter.create_counter("fallbacks"))
    return reader


def _points(reader, name):
    data = reader.get_metrics_data()
    return [
        point
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for metric in sm.metrics
        if metric.name == name
        for point in metric.data.data_points
    ]


class TestStage:
    def test_records_timing_span_and_duration(self, spans, metric_reader, capsys):
        timing = {}
        with telemetry.stage("atom_revision", timing, "Atom revision", atoms=3) as sp:
            sp.set_attribute("extra", "yes")

        assert timing["atom_revision"] >= 0
        assert "[FactReasoner][TIMING] Atom revision:" in capsys.readouterr().out
        (span,) = spans.get_finished_spans()
        assert span.name == "fact_reasoner.atom_revision"
        assert span.attributes["atoms"] == 3
        assert span.attributes["extra"] == "yes"
        (point,) = _points(metric_reader, "stage.duration")
        assert point.attributes == {"stage": "atom_revision"}

    def test_failed_stage_records_only_the_span(self, spans, capsys):
        timing = {}
        with pytest.raises(RuntimeError):
            with telemetry.stage("atom_revision", timing, "Atom revision"):
                raise RuntimeError("boom")

        assert timing == {}
        assert "TIMING" not in capsys.readouterr().out
        (span,) = spans.get_finished_spans()
        assert not span.status.is_ok

    def test_drops_unsupported_attribute_values(self, spans):
        with telemetry.span("s", keep=1, none=None, obj=object()):
            pass
        (span,) = spans.get_finished_spans()
        assert dict(span.attributes) == {"keep": 1}

    def test_noop_without_opentelemetry(self, monkeypatch):
        for name in ("_tracer", "_stage_duration", "_fallbacks"):
            monkeypatch.setattr(telemetry, name, None)
        timing = {}
        with telemetry.stage("x", timing) as sp:
            sp.set_attribute("ignored", 1)
        telemetry.count_fallback("reviser", "parse_error")
        assert "x" in timing


class TestTraced:
    def test_wraps_sync_and_async_functions(self, spans):
        @telemetry.traced("sync_op")
        def sync_op(x):
            return x + 1

        @telemetry.traced("async_op")
        async def async_op(x):
            return x * 2

        assert sync_op(1) == 2
        assert asyncio.run(async_op(2)) == 4
        assert [s.name for s in spans.get_finished_spans()] == ["sync_op", "async_op"]


class TestCountFallback:
    def test_counts_by_component_and_reason(self, metric_reader):
        telemetry.count_fallback("reviser", "parse_error")
        telemetry.count_fallback("reviser", "parse_error")
        telemetry.count_fallback("nli", "generation_error")

        counts = {
            (p.attributes["component"], p.attributes["reason"]): p.value
            for p in _points(metric_reader, "fallbacks")
        }
        assert counts == {("reviser", "parse_error"): 2, ("nli", "generation_error"): 1}


class TestRunCoroutineContext:
    def test_spans_in_background_loop_nest_under_caller(self, spans):
        """run_coroutine runs on another thread; the caller's span must carry over.

        asyncio provides this: run_coroutine_threadsafe schedules the task via
        call_soon_threadsafe, which copies the caller's context. The test guards
        it, since LLM-call spans would otherwise detach from their pipeline stage.
        """

        async def child():
            with telemetry.span("child"):
                await asyncio.sleep(0)

        with telemetry.span("parent"):
            run_coroutine(child())

        by_name = {s.name: s for s in spans.get_finished_spans()}
        assert by_name["child"].parent is not None
        assert by_name["child"].parent.span_id == by_name["parent"].context.span_id
        assert by_name["child"].context.trace_id == by_name["parent"].context.trace_id

    def test_caller_context_does_not_leak_into_the_loop(self, spans):
        """A later call without a parent span must start a fresh trace."""

        async def child(name):
            with telemetry.span(name):
                pass

        with telemetry.span("parent"):
            run_coroutine(child("first"))
        run_coroutine(child("second"))

        by_name = {s.name: s for s in spans.get_finished_spans()}
        assert by_name["second"].parent is None
