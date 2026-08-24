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

"""Load API credentials from a ``.env`` file into the process environment.

The backends read their credentials from environment variables --- ``RITSBackend``
does ``os.environ["RITS_API_KEY"]`` and raises ``KeyError`` if it is missing --- but
the repository keeps those values in a gitignored ``.env`` at the project root. A
script run from a shell that has not sourced that file therefore dies at backend
construction with an opaque ``KeyError``, several seconds into a run that looked
like it was starting fine.

:func:`load_dotenv` closes that gap. Call it once at the top of an entry point,
before any backend is built.

Two deliberate properties:

* **The real environment wins.** A variable already set in the process is never
  overwritten unless ``override=True``. So ``RITS_API_KEY=... python script.py``
  and a CI secret both keep working, and a stale ``.env`` cannot silently shadow
  the credential an operator just exported.
* **Values are never logged.** The return value and the summary line report only
  which *names* were set, never their contents, so a verbose run cannot leak a key
  into a terminal transcript or a CI log.

Parsing follows the usual ``.env`` conventions: ``KEY=value`` one per line, ``#``
comments, blank lines ignored, optional ``export`` prefix, and surrounding single
or double quotes stripped. It is implemented here rather than delegated to
``python-dotenv`` so that credential loading never depends on an optional package
being installed --- the failure mode that would produce is exactly the one this
module exists to prevent.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Default file name searched for.
DOTENV_NAME = ".env"

#: Setting this to a truthy value disables :func:`load_dotenv` entirely.
#:
#: Needed because loading is a *side effect on the process environment*, and a
#: caller that has deliberately cleared a credential must not have it silently
#: restored. Tests that assert "this command refuses to run without a key" are the
#: concrete case: without an opt-out, an entry point that loads .env makes that
#: precondition unachievable and the guard untestable.
DISABLE_VAR = "FACT_REASONER_NO_DOTENV"


def find_dotenv(start: str | os.PathLike[str] | None = None) -> Path | None:
    """Search upward from ``start`` for a ``.env`` file.

    Args:
        start: Directory to begin from. Defaults to this file's location, so the
            project root is found regardless of the caller's working directory ---
            which matters because scripts here are run both from the repo root and
            from subdirectories.

    Returns:
        The path to the first ``.env`` found, or None.
    """
    here = Path(start) if start is not None else Path(__file__).resolve().parent
    if here.is_file():
        here = here.parent
    for directory in [here, *here.parents]:
        candidate = directory / DOTENV_NAME
        if candidate.is_file():
            return candidate
    return None


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse ``.env`` contents into a mapping.

    Args:
        text: The file contents.

    Returns:
        ``{name: value}``. Malformed lines (no ``=``) are skipped rather than
        raising: a credential file with one stray line should still yield the keys
        it does define.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, sep, value = line.partition("=")
        if not sep:
            continue
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        # Strip one layer of matching quotes, the common .env convention.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[name] = value
    return values


def load_dotenv(
    path: str | os.PathLike[str] | None = None,
    *,
    override: bool = False,
    verbose: bool = False,
) -> list[str]:
    """Load ``.env`` into ``os.environ`` and return the names that were set.

    Args:
        path: Explicit ``.env`` path. When None, searches upward from this module
            for the project-root file.
        override: Whether to replace variables already present in the environment.
            False by default, so an explicitly exported credential always wins over
            the file.
        verbose: Print a one-line summary naming the variables set. Names only ---
            values are never printed.

    Setting :data:`DISABLE_VAR` in the environment disables loading entirely.

    Returns:
        The names actually set, in file order. Empty when no file was found or
        every name was already present.
    """
    if os.environ.get(DISABLE_VAR):
        if verbose:
            print(f"[env] {DISABLE_VAR} is set; not loading .env")
        return []

    dotenv_path = Path(path) if path is not None else find_dotenv()
    if dotenv_path is None or not dotenv_path.is_file():
        if verbose:
            print("[env] no .env file found")
        return []

    applied: list[str] = []
    for name, value in parse_dotenv(dotenv_path.read_text()).items():
        if not override and name in os.environ:
            continue
        os.environ[name] = value
        applied.append(name)

    if verbose:
        if applied:
            print(
                f"[env] loaded {len(applied)} variable(s) from "
                f"{dotenv_path}: {', '.join(applied)}"
            )
        else:
            print(f"[env] {dotenv_path} had nothing new to set")
    return applied


def require_env(*names: str, hint: str | None = None) -> None:
    """Raise a readable error if any of ``names`` is missing from the environment.

    Backends raise a bare ``KeyError`` for a missing credential, which does not say
    what to do about it. Calling this first turns that into an actionable message
    at the start of a run rather than partway through one.

    Args:
        names: Required environment variable names.
        hint: Extra guidance appended to the message.

    Raises:
        SystemExit: If any name is unset or empty.
    """
    missing = [n for n in names if not os.environ.get(n)]
    if not missing:
        return
    message = (
        f"Missing required environment variable(s): {', '.join(missing)}. "
        f"Add them to a .env file at the project root, or export them."
    )
    if hint:
        message = f"{message} {hint}"
    raise SystemExit(message)
