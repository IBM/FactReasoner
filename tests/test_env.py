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

"""Tests for the ``.env`` credential loader."""

from __future__ import annotations

import os

import pytest

from fact_reasoner.env import (
    find_dotenv,
    load_dotenv,
    parse_dotenv,
    require_env,
)


def test_parses_the_usual_conventions():
    parsed = parse_dotenv(
        """
        # a comment
        PLAIN=value
        export EXPORTED=exported-value
        QUOTED="double quoted"
        SINGLE='single quoted'
        SPACED  =  padded
        EMPTY=
        """
    )
    assert parsed["PLAIN"] == "value"
    assert parsed["EXPORTED"] == "exported-value"
    assert parsed["QUOTED"] == "double quoted"
    assert parsed["SINGLE"] == "single quoted"
    assert parsed["SPACED"] == "padded"
    assert parsed["EMPTY"] == ""
    assert "# a comment" not in parsed


def test_skips_malformed_lines_rather_than_raising():
    """One stray line must not cost us the keys the file does define."""
    parsed = parse_dotenv("GOOD=1\nthis line has no equals sign\nALSO_GOOD=2")
    assert parsed == {"GOOD": "1", "ALSO_GOOD": "2"}


def test_a_value_containing_equals_is_kept_whole():
    """Base64 and URL credentials routinely contain '='."""
    assert parse_dotenv("TOKEN=abc=def==")["TOKEN"] == "abc=def=="


def test_does_not_override_the_real_environment(tmp_path, monkeypatch):
    """An exported credential must beat the file.

    Otherwise a stale .env silently shadows the key an operator just exported,
    which is the hardest kind of credential bug to see.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_KEY=from-file\n")
    monkeypatch.setenv("MY_TEST_KEY", "from-environment")

    applied = load_dotenv(env_file)
    assert applied == []
    assert os.environ["MY_TEST_KEY"] == "from-environment"


def test_override_flag_replaces_the_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MY_TEST_KEY=from-file\n")
    monkeypatch.setenv("MY_TEST_KEY", "from-environment")

    assert load_dotenv(env_file, override=True) == ["MY_TEST_KEY"]
    assert os.environ["MY_TEST_KEY"] == "from-file"


def test_sets_missing_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("BRAND_NEW_KEY=hello\n")
    monkeypatch.delenv("BRAND_NEW_KEY", raising=False)

    assert load_dotenv(env_file) == ["BRAND_NEW_KEY"]
    assert os.environ["BRAND_NEW_KEY"] == "hello"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_verbose_never_prints_a_value(tmp_path, capsys, monkeypatch):
    """The summary names variables; it must not leak their contents."""
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_TEST_KEY=super-secret-value\n")
    monkeypatch.delenv("SECRET_TEST_KEY", raising=False)

    load_dotenv(env_file, verbose=True)
    out = capsys.readouterr().out
    assert "SECRET_TEST_KEY" in out
    assert "super-secret-value" not in out


def test_finds_the_project_root_dotenv():
    """The repo's own .env is discovered from the package location.

    Skipped when absent, since a fresh checkout has no credentials file.
    """
    found = find_dotenv()
    if found is None:
        pytest.skip("no .env in this checkout")
    assert found.name == ".env"
    assert found.is_file()


def test_require_env_passes_when_set(monkeypatch):
    monkeypatch.setenv("PRESENT_KEY", "x")
    require_env("PRESENT_KEY")  # must not raise


def test_require_env_reports_every_missing_name(monkeypatch):
    monkeypatch.delenv("ABSENT_ONE", raising=False)
    monkeypatch.delenv("ABSENT_TWO", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        require_env("ABSENT_ONE", "ABSENT_TWO", hint="Try the .env file.")
    message = str(excinfo.value)
    assert "ABSENT_ONE" in message and "ABSENT_TWO" in message
    assert "Try the .env file." in message


def test_require_env_treats_empty_as_missing(monkeypatch):
    """An empty string is not a usable credential."""
    monkeypatch.setenv("BLANK_KEY", "")
    with pytest.raises(SystemExit):
        require_env("BLANK_KEY")


def test_disable_var_suppresses_loading(tmp_path, monkeypatch):
    """The opt-out must win, so a cleared credential stays cleared.

    Loading .env mutates the process environment, so an entry point that loads it
    would otherwise make "this command refuses to run without a key" untestable ---
    and, worse, would silently restore a credential an operator had deliberately
    unset.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("SHOULD_NOT_LOAD=nope\n")
    monkeypatch.delenv("SHOULD_NOT_LOAD", raising=False)
    monkeypatch.setenv("FACT_REASONER_NO_DOTENV", "1")

    assert load_dotenv(env_file) == []
    assert "SHOULD_NOT_LOAD" not in os.environ


def test_disable_var_ignored_when_empty(tmp_path, monkeypatch):
    """Only a truthy value disables loading."""
    env_file = tmp_path / ".env"
    env_file.write_text("LOADS_FINE=yes\n")
    monkeypatch.delenv("LOADS_FINE", raising=False)
    monkeypatch.setenv("FACT_REASONER_NO_DOTENV", "")

    assert load_dotenv(env_file) == ["LOADS_FINE"]
