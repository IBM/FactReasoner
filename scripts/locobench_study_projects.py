"""Create one annotator's Label Studio projects from their bundle (local Label Studio).

Creates four projects -- practice task 1, practice task 2, task 1 (fillers), task 2 (pairs) --
or, if the bundle has a ``projects.json`` (a list of [title, config, tasks file]), those, with the bundle's labeling configs, sequential task order, no skipping and one annotation per
task, and imports the tasks. Uses the REST API, so it works with any Label Studio 1.x.

Authentication: Label Studio shows the token under *Account & Settings*. A legacy token is sent
as ``Authorization: Token <token>``; a personal access token (Label Studio 1.15+) is a refresh
token, exchanged here for an access token and sent as ``Bearer``. ``--scheme auto`` tries both.

    python scripts/locobench_study_projects.py --bundle path/to/annotator_3 \\
        --url http://localhost:8080 --token <your token>

Manual alternative (no script): see ``study/setup.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

PROJECTS = [  # (title, config, tasks file); Label Studio caps titles at 50 characters
    ("LoCoBench practice 1: fillers", "filler_reading.xml", "practice_task1.json"),
    ("LoCoBench practice 2: pairs", "rung_comparison.xml", "practice_task2.json"),
    ("LoCoBench task 1: fillers", "filler_reading.xml", "task1_fillers.json"),
    ("LoCoBench task 2: pairs", "rung_comparison.xml", "task2_pairs.json"),
]


def request(url, method="GET", body=None, headers=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:2000]
        raise SystemExit(f"{method} {url} -> HTTP {e.code}: {detail}") from None


def auth_header(url, token, scheme):
    """The Authorization header for a legacy token or a personal access (refresh) token."""
    if scheme in ("token", "auto"):
        h = {"Authorization": f"Token {token}"}
        try:
            request(f"{url}/api/current-user/whoami", headers=h)
            return h
        except SystemExit:
            if scheme == "token":
                raise
    acc = request(f"{url}/api/token/refresh", "POST", {"refresh": token})["access"]
    return {"Authorization": f"Bearer {acc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Your annotator_<k> folder.")
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--scheme", choices=("auto", "token", "bearer"), default="auto")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    h = auth_header(url, args.token, args.scheme)
    who = os.path.basename(os.path.normpath(args.bundle))
    # A bundle may list its own projects (the coherence study's bundles do); else the default.
    pj = os.path.join(args.bundle, "projects.json")
    projects = json.load(open(pj)) if os.path.exists(pj) else PROJECTS
    for title, cfg, tasks in projects:
        config = open(os.path.join(args.bundle, cfg)).read()
        proj = request(f"{url}/api/projects", "POST", {
            "title": f"{title} ({who})", "label_config": config,
            "sampling": "Sequential sampling", "show_skip_button": False,
            "maximum_annotations": 1, "show_instruction": False,
            "show_annotation_history": False, "enable_empty_annotation": False,
        }, h)
        data = json.load(open(os.path.join(args.bundle, tasks)))
        request(f"{url}/api/projects/{proj['id']}/import", "POST", data, h)
        print(f"created project {proj['id']}: {title} ({len(data)} tasks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
