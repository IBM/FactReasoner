"""Build the item files for the external checks (plan: .claude/plans/external-datasets-eval.md).

Reads raw downloads from data/ext/<name>/raw/ (gitignored) and writes data/ext/<name>/items.jsonl
plus a README recording the source, the licence and the SHA-256 of every raw file.

    .venv/bin/python scripts/ext_build.py              # all datasets
    .venv/bin/python scripts/ext_build.py contradoc    # one
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fact_reasoner.exteval import adapters as AD  # noqa: E402

SOURCES = {
    "loco-shuffle": ("LoCoBench slot3 test split (this repo), rung-0 items of the 180 non-ORDER "
                     "families", "this repository", None),
    "contradoc": ("ContraDoc, Li, Raheja & Kumar, NAACL 2024 -- "
                  "https://github.com/ddhruvkr/contradoc (ContraDoc.json)", "Apache-2.0", "raw"),
    "summeval": ("SummEval, Fabbri et al., TACL 2021 -- https://storage.googleapis.com/"
                 "sfr-summarization-repo-research/model_annotations.aligned.jsonl", "MIT", "raw"),
    "chatprotect": ("Muendler et al., ICLR 2024 -- https://github.com/eth-sri/ChatProtect "
                    "(annotations/glm/reconciled, annotations/verifiable_sc)", "Apache-2.0", "raw"),
    "qt30": ("QT30, Hautli-Janisz et al., LREC 2022 -- https://corpora.aifdb.org/api/"
             "archive.php?shortname=qt30&filetype=zip", "not stated on the corpus page; "
             "do not redistribute derived items", "raw"),
    "gdtb": ("GDTB (PDTB-3 relations of GUM), Liu et al., EMNLP 2024, via UD_English-GUM "
             "en_gum-ud-{dev,test}.conllu (commit 8e168382a2)", "CC BY-NC-SA 4.0 (GUM; "
             "Reddit documents excluded)", "raw"),
}


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(name: str) -> list[dict]:
    raw = os.path.join(REPO, "data", "ext", name, "raw")
    if name == "loco-shuffle":
        return AD.loco_shuffles(REPO)
    return {"contradoc": AD.contradoc, "summeval": AD.summeval, "chatprotect": AD.chatprotect,
            "qt30": AD.qt30, "gdtb": AD.gdtb}[name](raw)


def main() -> int:
    names = sys.argv[1:] or list(SOURCES)
    for name in names:
        items = build(name)
        d = os.path.join(REPO, "data", "ext", name)
        AD.write(os.path.join(d, "items.jsonl"), items)
        src, lic, raw = SOURCES[name]
        lines = [f"# data/ext/{name}", "", f"Source: {src}", "", f"Licence: {lic}", "",
                 f"Built by `scripts/ext_build.py {name}` (seed {AD.SEED}); adapter in "
                 "`src/fact_reasoner/exteval/adapters.py`.", ""]
        if raw:
            lines += ["Raw files (gitignored, `raw/`), SHA-256:", ""]
            for root, _, files in sorted(os.walk(os.path.join(d, raw))):
                for f in sorted(files):
                    p = os.path.join(root, f)
                    if name == "qt30" and "nodesets" in p:
                        continue  # hashed through the archive they came from
                    lines.append(f"- `{os.path.relpath(p, d)}` `{sha(p)}`")
            lines.append("")
        s = AD.summary(items)
        lines += ["Items: " + json.dumps(s), ""]
        open(os.path.join(d, "README.md"), "w").write("\n".join(lines))
        print(f"{name}: {json.dumps(s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
