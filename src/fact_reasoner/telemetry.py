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

# OpenTelemetry instrumentation for the FactReasoner pipeline.
#
# Pipeline stages become spans, so the per-call spans Mellea emits for every LLM
# request nest under the stage that issued them, and component fallbacks become
# a counter. Everything here is opt-in and costs nothing when off:
#
#   * Without the OpenTelemetry packages (`pip install "fact_reasoner[telemetry]"`)
#     every helper is a no-op.
#   * With them, nothing is exported until Mellea's environment variables enable
#     it -- e.g. MELLEA_TRACES_ENABLED=true MELLEA_TRACES_CONSOLE=true, and
#     MELLEA_METRICS_ENABLED=true MELLEA_METRICS_CONSOLE=true. See the README.
#
# We deliberately reuse Mellea's configuration instead of adding our own:
# Mellea installs the global tracer/meter providers when `mellea.telemetry` is
# imported, so importing it first means our spans share its exporters and end
# up in the same traces as its LLM-call spans.

import functools
import inspect
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

try:
    import mellea.telemetry  # noqa: F401  (side effect: configures providers)
except ImportError:  # pragma: no cover - mellea is a hard dependency
    pass

try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
except ImportError:
    _otel_metrics = None
    _otel_trace = None

_SCOPE = "fact_reasoner"

if _otel_trace is not None and _otel_metrics is not None:
    _tracer = _otel_trace.get_tracer(_SCOPE)
    _meter = _otel_metrics.get_meter(_SCOPE)
    _stage_duration = _meter.create_histogram(
        "fact_reasoner.stage.duration",
        unit="s",
        description="Wall-clock duration of a FactReasoner pipeline stage",
    )
    _fallbacks = _meter.create_counter(
        "fact_reasoner.fallbacks",
        description=(
            "Component results replaced by a default (e.g. an unrevised atom or "
            "a neutral NLI label) after a generation or parsing failure"
        ),
    )
else:
    _tracer = None
    _stage_duration = None
    _fallbacks = None


def _attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    """Keep only values OpenTelemetry accepts as span/metric attributes."""
    return {
        k: v for k, v in attributes.items() if isinstance(v, (str, bool, int, float))
    }


class _NoSpan:
    """Stand-in span yielded when OpenTelemetry is not installed."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Run the enclosed block inside a span named ``name``.

    Yields the span so callers can attach attributes discovered while the block
    runs (``sp.set_attribute(...)``); without OpenTelemetry it is a no-op object.
    Attribute values that are not ``str``/``bool``/``int``/``float`` (including
    ``None``) are dropped.
    """
    if _tracer is None:
        yield _NoSpan()
        return
    with _tracer.start_as_current_span(name, attributes=_attributes(attributes)) as s:
        yield s


@contextmanager
def stage(
    name: str,
    timing: dict[str, float] | None = None,
    label: str | None = None,
    **attributes: Any,
) -> Iterator[Any]:
    """Time a pipeline stage and trace it as a ``fact_reasoner.<name>`` span.

    On success the duration is stored in ``timing[name]`` (when given), printed as
    a ``[FactReasoner][TIMING] <label>`` line (when ``label`` is given), and
    recorded in the ``fact_reasoner.stage.duration`` histogram. A stage that
    raises records nothing but the span, which carries the exception.
    """
    start = time.perf_counter()
    with span(f"{_SCOPE}.{name}", **attributes) as s:
        yield s
    seconds = time.perf_counter() - start
    if timing is not None:
        timing[name] = seconds
    if label is not None:
        print(f"[FactReasoner][TIMING] {label}: {seconds:.4f}s")
    if _stage_duration is not None:
        _stage_duration.record(seconds, {"stage": name})


def traced(name: str) -> Callable:
    """Decorator: run a sync or async function inside a span named ``name``."""

    def decorate(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with span(name):
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorate


def count_fallback(component: str, reason: str) -> None:
    """Count one result replaced by a default after a failure.

    Args:
        component: The component that fell back (e.g. ``"reviser"``, ``"nli"``).
        reason: Why: ``"generation_error"``, ``"sampling_failed"`` or
            ``"parse_error"`` (or a component-specific reason).
    """
    if _fallbacks is not None:
        _fallbacks.add(1, {"component": component, "reason": reason})
