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

import asyncio
import json
import re
import threading
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

import requests
import tqdm

from fact_reasoner import telemetry

LOOP_BUDGET = 5

# Throttling defaults for batched LLM generation. The rate limit protects
# against provider rate-limit (429) errors, while the concurrency ceiling bounds
# the number of in-flight requests (and thus open sockets) at any given time.
MAX_REQUESTS_PER_MINUTE = 1500
MAX_CONCURRENT_REQUESTS = 32


class AsyncRateLimiter:
    """Token-bucket rate limiter for asyncio.

    Allows at most ``rate`` acquisitions per ``period`` seconds, smoothing bursts
    by refilling tokens continuously rather than in fixed windows. Safe to share
    across concurrent coroutines running on the same event loop.
    """

    def __init__(self, rate: int, period: float = 60.0):
        """
        Args:
            rate: int
                Maximum number of acquisitions allowed per ``period`` seconds.
            period: float
                The length of the rate-limiting window, in seconds.
        """
        if rate <= 0:
            raise ValueError("rate must be a positive integer")
        if period <= 0:
            raise ValueError("period must be a positive number of seconds")

        self._rate = rate
        self._period = period
        self._allowance = float(rate)  # available tokens
        self._last = None  # lazily initialized to the event-loop clock on first use
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until a token is available, then consume it."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if self._last is None:
                self._last = now

            # Refill tokens proportionally to the elapsed time.
            self._allowance += (now - self._last) * (self._rate / self._period)
            self._last = now
            if self._allowance > self._rate:
                self._allowance = float(self._rate)  # never accumulate beyond capacity

            if self._allowance < 1.0:
                # Not enough budget yet; sleep until a single token is available.
                sleep_for = (1.0 - self._allowance) * (self._period / self._rate)
                await asyncio.sleep(sleep_for)
                # We slept exactly long enough to earn and consume one token.
                # Advance the clock past the sleep so the next caller does not
                # re-credit the elapsed sleep time as additional tokens.
                self._allowance = 0.0
                self._last = asyncio.get_event_loop().time()
            else:
                self._allowance -= 1.0


class BackendThrottle:
    """Concurrency ceiling and rate limit shared by every batch sent to a backend.

    :func:`run_throttled` on its own bounds only the batch it is given, so two
    stages (or two dataset items) running at the same time each get the full
    budget and together overload the server. Passing the same throttle to every
    batch makes the budget global to the backend.

    Pipeline stages run on more than one event loop (``asyncio.run`` for
    ``build()``, the background loop of ``run_coroutine`` for the rest), and
    asyncio primitives are bound to one loop. The state here is guarded by a
    ``threading.Lock`` instead, and waiters poll with ``asyncio.sleep``, so one
    throttle is safe to share across loops and threads.
    """

    # How long a waiter sleeps between checks for a free slot or token.
    _POLL_SECONDS = 0.01

    def __init__(
        self,
        max_concurrency: int = MAX_CONCURRENT_REQUESTS,
        rate_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    ):
        """
        Args:
            max_concurrency: int
                Maximum number of items in flight at once, across all batches.
            rate_per_minute: int
                Maximum number of items started per minute, across all batches.
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")
        if rate_per_minute <= 0:
            raise ValueError("rate_per_minute must be a positive integer")

        self.max_concurrency = max_concurrency
        self.rate_per_minute = rate_per_minute
        self._in_flight = 0
        self._tokens = float(rate_per_minute)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def _try_acquire(self) -> bool:
        """Take a slot and a token if both are available, without blocking."""
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                float(self.rate_per_minute),
                self._tokens + (now - self._last) * self.rate_per_minute / 60.0,
            )
            self._last = now
            if self._in_flight >= self.max_concurrency or self._tokens < 1.0:
                return False
            self._in_flight += 1
            self._tokens -= 1.0
            return True

    async def acquire(self) -> None:
        """Wait until a slot and a rate token are free, then take both."""
        while not self._try_acquire():
            await asyncio.sleep(self._POLL_SECONDS)

    def release(self) -> None:
        """Return the slot taken by :meth:`acquire`."""
        with self._lock:
            self._in_flight -= 1


# One throttle per backend object, so every component built on the same backend
# shares a single budget. Weak keys let a discarded backend take its throttle
# with it.
_backend_throttles: "weakref.WeakKeyDictionary[Any, BackendThrottle]" = (
    weakref.WeakKeyDictionary()
)
_backend_throttles_lock = threading.Lock()


def throttle_for(backend: Any) -> BackendThrottle | None:
    """Return the shared :class:`BackendThrottle` for ``backend``.

    Created with the module defaults on first use. Returns None for a backend
    that cannot be weakly referenced (plain stand-ins in tests), in which case
    callers fall back to per-batch throttling.
    """
    with _backend_throttles_lock:
        try:
            throttle = _backend_throttles.get(backend)
            if throttle is None:
                throttle = BackendThrottle()
                _backend_throttles[backend] = throttle
        except TypeError:
            return None
        return throttle


async def run_throttled(
    factory: Callable[[Any], Awaitable[Any]],
    items: list[Any],
    *,
    max_concurrency: int = MAX_CONCURRENT_REQUESTS,
    rate_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    on_progress: Callable[[], None] | None = None,
    throttle: BackendThrottle | None = None,
) -> list[Any]:
    """Run one coroutine per item with bounded concurrency and rate limiting.

    A fresh coroutine is created for each item via ``factory`` right before it
    runs, so the rate limiter gates *when* each generation starts. Every item is
    awaited independently: if one raises, the exception is captured and returned
    in place instead of propagating, so a single failure never drops the
    remaining results.

    Args:
        factory: Callable[[item], Awaitable]
            Builds a fresh coroutine for a single item. Called once per item.
        items: List[Any]
            The inputs to process.
        max_concurrency: int
            Maximum number of coroutines running at any given time.
        rate_per_minute: int
            Maximum number of coroutines started per minute.
        on_progress: Optional[Callable[[], None]]
            If given, called once (with no arguments) each time an item's
            coroutine completes — succeeds or fails. Fires in completion order,
            not input order, so it is suitable for driving a progress bar. Must
            not block.
        throttle: Optional[BackendThrottle]
            A budget shared with every other batch sent to the same backend
            (see :func:`throttle_for`). When given, it replaces the per-batch
            ``max_concurrency`` / ``rate_per_minute`` limits. Both count items,
            not requests: an item that issues several generations (SIMBA-UQ)
            holds one slot.

    Returns:
        List[Any]: Results positionally aligned with ``items``. For any item
        whose coroutine raised, the corresponding entry is the ``Exception``
        object (never reordered, never dropped).
    """
    if throttle is None:
        limiter = AsyncRateLimiter(rate_per_minute)
        sem = asyncio.Semaphore(max_concurrency)

    async def _call(item: Any) -> Any:
        try:
            return await factory(item)
        except Exception as e:  # capture, so sibling requests are not dropped  # noqa: BLE001
            return e
        finally:
            if on_progress is not None:
                on_progress()

    async def _one(item: Any) -> Any:
        if throttle is not None:
            await throttle.acquire()
            try:
                return await _call(item)
            finally:
                throttle.release()
        async with sem:
            await limiter.acquire()
            return await _call(item)

    tasks = [asyncio.create_task(_one(item)) for item in items]
    # _one never raises, so gather returns one result per item in order.
    return await asyncio.gather(*tasks)


def handle_generation_error(
    error: Exception, component: str, message: str, raise_errors: bool
) -> None:
    """Report a failed LLM call, or re-raise it when the caller opted in.

    By default the error is printed as ``"{message}: {error}"`` and counted as a
    ``generation_error`` fallback; the calling stage then substitutes its
    default result. With ``raise_errors`` the original exception is re-raised
    as is (same object, type, traceback and cause), so callers can catch it.

    Args:
        error: Exception
            The exception raised by the LLM call.
        component: str
            The component name used for telemetry (e.g. ``"atomizer"``).
        message: str
            The prefix of the printed line (e.g. ``"[Atomizer] Batch item failed"``).
        raise_errors: bool
            If True, re-raise ``error`` instead of reporting it.

    Raises:
        Exception: ``error`` itself, when ``raise_errors`` is True.
    """
    if raise_errors:
        raise error
    print(f"{message}: {error}")
    telemetry.count_fallback(component, "generation_error")


async def gather_with_progress(
    coros: list[Awaitable[Any]],
    *,
    on_progress: Callable[[], None] | None = None,
) -> list[Any]:
    """Await already-built coroutines concurrently, with a per-completion hook.

    Like :func:`asyncio.gather` (results are returned in **input order**), but
    calls ``on_progress`` once as each coroutine completes — so a progress bar
    advances as work finishes rather than all at once at the barrier. Unlike
    :func:`run_throttled`, this takes pre-built coroutines (no factory / rate
    limiting): use it to add progress to an existing ``asyncio.gather`` call.

    Args:
        coros: The coroutines/awaitables to run concurrently.
        on_progress: If given, called with no arguments each time one completes
            (in completion order, not input order). Must not block. Exceptions
            propagate as they would from ``asyncio.gather`` (default behavior).

    Returns:
        Results positionally aligned with ``coros``.
    """

    async def _indexed(index: int, coro: Awaitable[Any]):
        result = await coro
        if on_progress is not None:
            on_progress()
        return index, result

    tasks = [asyncio.ensure_future(_indexed(i, c)) for i, c in enumerate(coros)]
    results: list[Any] = [None] * len(tasks)
    for completed in asyncio.as_completed(tasks):
        index, result = await completed
        results[index] = result
    return results


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# String manipulation utils
def join_segments(*args: str | list[str], separator: str = "\n\n\n") -> str:
    """Joins an unspecified number of strings using the separator."""
    all_segments = []

    for arg in args:
        if isinstance(arg, list):
            all_segments.extend(arg)
        else:
            all_segments.append(strip_string(str(arg)))

    return strip_string(separator.join(all_segments))


def strip_string(s: str) -> str:
    """Strips a string of newlines and spaces."""
    return s.strip(" \n")


def punctuation_only_inside_quotes(text):
    # find all quoted sections (single or double quotes)
    quoted_spans = [match.span() for match in re.finditer(r'"[^"]*"|\'[^\']*\'', text)]

    def is_inside_quotes(index):
        return any(start < index < end for start, end in quoted_spans)

    # check each comma and semicolon
    for i, char in enumerate(text):
        if char in [",", ";"] and not is_inside_quotes(i):
            return False  # found punctuation outside quotes
    return True


def extract_first_square_brackets(input_string: str) -> str:
    """Extracts the contents of the FIRST string between square brackets."""
    raw_result = re.findall(r"\[.*?\]", input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[0][1:-1]
    else:
        return ""


def extract_last_square_brackets(input_string: str) -> str:
    """Extracts the contents of the LAST string between square brackets.

    Symmetric counterpart to :func:`extract_first_square_brackets`: returns the
    raw contents of the last ``[...]`` pair, with surrounding whitespace and any
    trailing punctuation stripped (so labels like ``[entailment.]`` normalize to
    ``entailment``).

    If no brackets are present, falls back to scanning for a bare NLI label word
    (``neutral``/``entailment``/``contradiction``) so the NLI path stays robust
    when the LLM drops the brackets entirely.
    """
    raw_result = re.findall(r"\[.*?\]", input_string, flags=re.DOTALL)
    if raw_result:
        return raw_result[-1][1:-1].strip().rstrip(".!?").strip()

    # Fallback: scan input for any bare NLI label word (handles cases where
    # the LLM drops the brackets entirely).
    words = re.findall(
        r"\b(neutral|entailment|contradiction)\b", input_string, flags=re.IGNORECASE
    )
    if words:
        return words[-1].lower()

    return ""


# Matches a JSON-style NLI verdict `"label": "<value>"` (tolerant of surrounding
# prose, code fences and extra keys). Group 1 is the label value.
_NLI_JSON_LABEL_RE = re.compile(r'"label"\s*:\s*"([^"]+)"')


def extract_nli_label_and_span(
    input_string: str,
) -> tuple[str, tuple[int, int] | None]:
    """Extract the NLI label and the character span of the label text.

    Auto-detects two output formats, in priority order:

    1. **JSON** — a ``{"label": "<value>"}`` object (or a bare ``"label": "..."``
       pair), tolerant of code fences, surrounding prose and extra keys. The last
       occurrence wins. The returned span covers the ``<value>`` text.
    2. **Brackets** — the last ``[...]`` pair (matching
       :func:`extract_last_square_brackets`). The span covers the bracket
       *interior*.
    3. **Bare word** — a bare ``neutral``/``entailment``/``contradiction`` word
       (last occurrence). The span covers that word.

    The span lets callers (e.g. the NLI logprobs probability) align token-level
    logprobs to exactly the label text this function reports, so the label and
    its probability can never disagree.

    Args:
        input_string: The raw model output text.

    Returns:
        ``(label, span)`` where ``label`` is lower-cased and stripped (``""`` if
        none found), and ``span`` is a ``(start, end)`` char range into
        ``input_string`` for the label text (``None`` if no label found).
    """
    # 1. JSON: {"label": "..."} — last match wins.
    json_matches = list(_NLI_JSON_LABEL_RE.finditer(input_string))
    if json_matches:
        m = json_matches[-1]
        return m.group(1).strip().lower(), m.span(1)

    # 2. Brackets: last [...] pair; span is the interior.
    bracket_matches = list(re.finditer(r"\[.*?\]", input_string, flags=re.DOTALL))
    if bracket_matches:
        m = bracket_matches[-1]
        start, end = m.span()
        interior = input_string[start + 1 : end - 1]
        # Mirror extract_last_square_brackets normalization for the label text.
        label = interior.strip().rstrip(".!?").strip()
        return label.lower(), (start + 1, end - 1)

    # 3. Bare NLI label word — last occurrence.
    word_matches = list(
        re.finditer(
            r"\b(neutral|entailment|contradiction)\b",
            input_string,
            flags=re.IGNORECASE,
        )
    )
    if word_matches:
        m = word_matches[-1]
        return m.group(1).lower(), m.span(1)

    return "", None


def extract_last_wrapped_response(input_string: str) -> str:
    """Extracts the contents of the LAST string between pairs of ###."""
    raw_result = re.findall(r"###.*?###", input_string, flags=re.DOTALL)

    if raw_result:
        return raw_result[-1][3:-3]
    else:
        return ""


def extract_first_code_block(input_string: str, ignore_language: bool = False) -> str:
    """Extracts the contents of a string between the first code block (```)."""
    if ignore_language:
        pattern = re.compile(r"```(?:\w+\n)?(.*?)```", re.DOTALL)
    else:
        pattern = re.compile(r"```(.*?)```", re.DOTALL)

    match = pattern.search(input_string)
    return strip_string(match.group(1)) if match else ""


def extract_logprobs_from_output(output: dict[str, Any]) -> list[Any]:
    """
    Extract the per-token log probabilities from the output metadata.

    Returns the backend's token-level logprobs as a list of ``{"token", "logprob"}``
    entries.

    Note: no tokens are dropped, and whether the list ends with a stop token is
    backend-dependent. OpenAI/vLLM ``content`` arrays usually hold only emitted
    content tokens, but some backends (e.g. a Bedrock imported model via litellm)
    append an end-of-turn token such as ``<|eot_id|>``. Callers that read the last
    token must handle both; NLI label extraction is unaffected because it locates
    the label by its brackets.

    Args:
        output: The output object containing the metadata with log probabilities.

    Returns:
        A list of per-token logprob entries extracted from the output.
    """

    raw = getattr(output, "raw", None)
    raw_response = raw.response if raw is not None else None
    logprobs_object = (
        raw_response.get("choices", [{}])[0].get("logprobs")
        if isinstance(raw_response, dict)
        else None
    )

    assert logprobs_object is not None, (
        "logprobs missing from response. Ensure the backend supports logprobs."
    )
    if "content" not in logprobs_object:
        raise ValueError(
            "logprobs object missing 'content' key. Check backend response format."
        )
    return logprobs_object["content"]


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch


# Google Drive related


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def strip_code_fences(s: str) -> str:
    """
    Strip markdown code fences from a string if present.

    Args:
        s: str
            The input string.
    Returns:
        str: The string without code fences.
    """

    s = s.strip()

    # Try a strict fenced block: ```json\n ... \n```
    m = re.match(r"^```(?:json|JSON)?\s*\n(.*?)\n```$", s, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: starts with ``` but may have irregular spacing/line breaks
    if s.startswith("```"):
        lines = s.splitlines()
        # Remove the opening fence line
        content_lines = lines[1:]
        # If the last line is a closing fence, drop it
        if content_lines and content_lines[-1].strip().startswith("```"):
            content_lines = content_lines[:-1]
        return "\n".join(content_lines).strip()

    # No fences detected; return as-is
    return s


def normalize_ws(text: str) -> str:
    """
    - Collapse all runs of whitespace (spaces, tabs, newlines) into a single space.
    - Escape inner pairs of double quotes:  ""phrase""  ->  \"phrase\"
      (Only when the pair of quotes is not already escaped.)

    Examples:
        Input:  'He said  ""Hello  world"" \n  today.'
        Output: 'He said \\"Hello world\\" today.'
    """
    if text is None:
        return text  # or raise ValueError("text must not be None")

    # Collapse all whitespace chunks to a single space
    # This turns tabs/newlines into spaces as well.
    collapsed = re.sub(r"\s+", " ", text).strip()
    collapsed = collapsed.replace("\n", "")

    return collapsed


def validate_json_code_block(
    input_string: str, required_keys: list[str] | None = None
) -> bool:
    """
    Checks if the input string is a valid JSON dictionary.

    Args:
        input_string: str
            The string to check.
        required_keys: List[str]
            List of keys that must be present in the JSON dictionary.

    Returns:
        bool: True if valid JSON, False otherwise. If required_keys is provided,
        then also checks if it is a dictionary and contains the required keys.
    """
    try:
        # Remove markdown fences if present
        cleaned = strip_code_fences(input_string)
        cleaned = normalize_ws(cleaned)

        # Attempt to parse the string as JSON
        data = json.loads(cleaned)

        # Check if it's a dictionary and has required keys
        if isinstance(data, dict) and required_keys:
            for key in required_keys:
                if key not in data:
                    return False
        return True
    except json.JSONDecodeError as e:
        # If parsing fails, it's not valid JSON
        print(f"Malformed JSON string: {e}")
        return False


def validate_markdown_code_block(input_string: str) -> bool:
    """
    Checks if the input string is a valid markdown code block.

    Args:
        input_string: str
            The string to check.

    Returns:
        bool: True if valid markdown code block, False otherwise.
    """

    return bool(input_string.strip().startswith("```") and input_string.strip().endswith("```"))
