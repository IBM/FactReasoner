"""The three-vendor committee on the IBM LiteLLM gateway, with robust JSON-list parsing.

Shared by the external checks; the same models as the LoCoBench committee. Every call retries
on network errors and on unparseable output; a list whose quote strings break JSON is
recovered object by object (only ``id`` and ``label`` are used).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

GATEWAY = "https://ete-litellm.ai-models.vpc.res.ibm.com/v1"
COMMITTEE = ("azure/gpt-6.1-sol", "gcp/gemini-3.1-pro-preview", "aws/claude-opus-5-5")


def ask_list(model: str, prompt: str, tries: int = 4) -> list[dict]:
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    client = OpenAI(base_url=GATEWAY, api_key=key, timeout=600)
    last = None
    for _ in range(tries):
        try:
            msg = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}])
            text = re.sub(r"```(?:json)?", "", msg.choices[0].message.content or "")
            m = re.search(r"\[.*\]", text, re.S)
            try:
                out = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                out = None
            if not isinstance(out, list):
                out = [{"id": i, "label": lab} for i, lab in re.findall(
                    r'"id"\s*:\s*"([^"]+)"\s*,\s*"label"\s*:\s*"([a-z_]+)"', text)] or None
            if isinstance(out, list):
                return out
            last = f"no JSON list (finish {msg.choices[0].finish_reason})"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    raise RuntimeError(f"{model}: {last}")


def run_jobs(jobs, cache_path: str, workers: int = 12) -> dict:
    """``jobs`` = [(key, model, prompt)]. Answers are appended to ``cache_path`` (jsonl) and
    reused on re-runs. Returns {(key, model): list}."""
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            if line.strip():
                r = json.loads(line)
                cache[(r["key"], r["model"])] = r["out"]
    todo = [j for j in jobs if (j[0], j[1]) not in cache]

    def run(job):
        key, model, prompt = job
        try:
            return key, model, ask_list(model, prompt)
        except RuntimeError as e:
            print(f"  FAILED {key} {model}: {e}")
            return key, model, None

    with ThreadPoolExecutor(max_workers=workers) as ex, open(cache_path, "a") as f:
        for key, model, out in ex.map(run, todo):
            if out is not None:
                cache[(key, model)] = out
                f.write(json.dumps({"key": key, "model": model, "out": out}) + "\n")
                f.flush()
    return cache


def majority(labels: list[str | None], allowed) -> str:
    top = Counter(v for v in labels if v in allowed).most_common(1)
    return top[0][0] if top and top[0][1] >= 2 else "unresolved"
