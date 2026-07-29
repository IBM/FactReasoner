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

"""Training utilities for the SIMBA-UQ *classifier* confidence method, specialized
to the NLI classification use case.

The SIMBA-UQ classifier maps a sample's pairwise-similarity feature vector to
``P(correct)``. Training it needs labeled *groups* of generated samples — one
group per query, each of size ``N = len(temperatures) * n_per_temp``, with a 0/1
correctness label per sample. This module produces exactly that from a labeled
NLI dataset of ``{premise, hypothesis, label}`` triples:

1. For each ``(premise, hypothesis, gold_label)`` pair, run the **same** SIMBA-UQ
   generation used at inference (the ``INSTRUCTION_NLI`` prompt under
   :class:`SIMBAUQSamplingStrategy`) to obtain ``N`` sample strings.
2. Extract each sample's NLI label and mark it correct (``1``) iff it equals the
   gold label, else ``0``.
3. Persist the labeled groups to a resumable JSONL cache, then fit + save a
   classifier from them (reusing the strategy's own feature extraction so the
   features match inference exactly).

The generation step is expensive (``N`` LLM calls per pair), so it is decoupled
from the cheap fit step via the JSONL cache; you can re-tune classifier
hyper-parameters without re-generating.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import mellea.stdlib.functional as mfuncs
import numpy as np
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext

# Local imports. NOTE: ``INSTRUCTION_NLI`` from ``fact_reasoner.core.nli`` is
# imported lazily inside ``generate_training_samples`` to avoid a circular import
# (``core.nli`` imports the ``uncertainty`` package, which imports this module).
from fact_reasoner.uncertainty.simbauq import (
    ProbabilisticClassifier,
    SIMBAUQSamplingStrategy,
)
from fact_reasoner.utils import extract_nli_label_and_span, run_throttled

# The NLI labels the classifier's correctness signal is defined over.
NLI_LABELS = ("entailment", "contradiction", "neutral")


def _sample_label(sample_text: str) -> str:
    """Extract the NLI label from a single generated sample.

    Uses the same primitive the NLIExtractor uses at inference
    (``extract_nli_label_and_span``, lower-cased), so the correctness signal is
    defined identically and handles both the JSON (``{"label": "..."}``) and
    bracket (``[...]``) output formats.
    """
    label, _ = extract_nli_label_and_span(str(sample_text))
    return label


def load_nli_pairs(
    path: str,
    *,
    num_pairs: int | None = None,
    balanced: bool = True,
    seed: int = 0,
) -> list[dict[str, str]]:
    """Load ``{premise, hypothesis, label}`` triples from a JSON array file.

    Args:
        path: Path to a JSON file containing a list of objects, each with
            ``premise``, ``hypothesis`` and ``label`` keys (label one of
            entailment / contradiction / neutral).
        num_pairs: If set, return at most this many pairs. Generation cost scales
            with the number of pairs, so this is the main knob for keeping a run
            affordable.
        balanced: When ``num_pairs`` is set, draw an (approximately) equal number
            of pairs per label rather than a uniform random subset.
        seed: RNG seed for the subset selection (deterministic).

    Returns:
        A list of ``{"premise", "hypothesis", "label"}`` dicts.

    Raises:
        ValueError: If the file does not contain a list, or an item is missing a
            required key, or a label is not a known NLI label.
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path!r}, got {type(data).__name__}.")  # noqa: TRY004

    pairs: list[dict[str, str]] = []
    for i, item in enumerate(data):
        try:
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = str(item["label"]).lower()
        except (KeyError, TypeError) as e:
            raise ValueError(f"Item {i} in {path!r} is missing a required key: {e}")
        if label not in NLI_LABELS:
            raise ValueError(
                f"Item {i} in {path!r} has unknown label {label!r} "
                f"(expected one of {list(NLI_LABELS)})."
            )
        pairs.append({"premise": premise, "hypothesis": hypothesis, "label": label})

    if num_pairs is None or num_pairs >= len(pairs):
        return pairs

    rng = random.Random(seed)
    if not balanced:
        return rng.sample(pairs, num_pairs)

    # Balanced subset: split the budget across the labels as evenly as possible.
    by_label: dict[str, list[dict[str, str]]] = {lbl: [] for lbl in NLI_LABELS}
    for p in pairs:
        by_label[p["label"]].append(p)

    per_label = num_pairs // len(NLI_LABELS)
    remainder = num_pairs - per_label * len(NLI_LABELS)
    selected: list[dict[str, str]] = []
    for idx, lbl in enumerate(NLI_LABELS):
        take = per_label + (1 if idx < remainder else 0)
        bucket = by_label[lbl]
        selected.extend(rng.sample(bucket, min(take, len(bucket))))
    rng.shuffle(selected)
    return selected


def _pair_key(premise: str, hypothesis: str) -> str:
    """Stable dedup key for a (premise, hypothesis) pair."""
    return f"{premise}␟{hypothesis}"


def _read_completed_keys(out_path: str) -> set:
    """Read the (premise, hypothesis) keys already present in a samples JSONL.

    Enables resuming a generation run: pairs already written are skipped.
    """
    completed: set = set()
    if not os.path.exists(out_path):
        return completed
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            completed.add(_pair_key(rec["premise"], rec["hypothesis"]))
    return completed


async def generate_training_samples(
    pairs: list[dict[str, str]],
    backend: Backend,
    out_path: str,
    *,
    temperatures: list[float] | None = None,
    n_per_temp: int = 5,
    similarity_metric: str = "rouge",
    num_workers: int = 4,
    progress: bool = False,
) -> dict[str, int]:
    """Generate SIMBA-UQ sample groups for each NLI pair and write them to JSONL.

    For every pair, runs ``INSTRUCTION_NLI`` under a :class:`SIMBAUQSamplingStrategy`
    (the exact inference path) and collects all ``N = len(temperatures)*n_per_temp``
    generated sample strings, labeling each ``1`` if its extracted NLI label equals
    the gold label else ``0``. Each pair becomes one JSONL line::

        {"premise", "hypothesis", "gold", "samples": [str×N], "labels": [0/1×N]}

    Only *complete* groups (exactly ``N`` parsable samples) are written — a partial
    group cannot be turned into fixed-width classifier features. The run is
    resumable: pairs already present in ``out_path`` are skipped.

    Args:
        pairs: Labeled NLI pairs (see :func:`load_nli_pairs`).
        backend: Mellea backend used for generation.
        out_path: Destination JSONL path (appended to).
        temperatures: SIMBA-UQ temperature schedule (default matches the strategy).
        n_per_temp: Samples per temperature.
        similarity_metric: Recorded for provenance; does not affect generation
            (similarity is only computed at fit/inference time).
        num_workers: Max concurrent pair generations.
        progress: If True, show a ``tqdm`` progress bar that advances as each
            pair's generation completes.

    Returns:
        A summary dict: ``{"written", "skipped_existing", "dropped_incomplete"}``.
    """
    # Lazy import to avoid a circular import at module load time (see note above).
    from fact_reasoner.core.nli import INSTRUCTION_NLI

    strategy = SIMBAUQSamplingStrategy(
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric=similarity_metric,
        confidence_method="aggregation",  # generation only; no classifier needed
    )
    expected_n = len(strategy.temperatures) * strategy.n_per_temp

    completed = _read_completed_keys(out_path)
    todo = [p for p in pairs if _pair_key(p["premise"], p["hypothesis"]) not in completed]
    skipped_existing = len(pairs) - len(todo)

    def factory(pair: dict[str, str]):
        return mfuncs.ainstruct(
            INSTRUCTION_NLI,
            context=SimpleContext(),
            backend=backend,
            user_variables={
                "premise_text": pair["premise"],
                "hypothesis_text": pair["hypothesis"],
            },
            strategy=strategy,
            return_sampling_results=True,
        )

    print(
        f"[nli-training] Generating samples for {len(todo)} pairs "
        f"({skipped_existing} already done); N={expected_n} samples/pair ..."
    )
    # Drive a progress bar from run_throttled's per-completion callback so the
    # bar advances as generations finish (not all at once at the end).
    bar = None
    on_progress = None
    if progress and todo:
        from tqdm import tqdm

        bar = tqdm(total=len(todo), desc="Generating", unit="pair")
        on_progress = bar.update  # tqdm.update() advances by 1 when called with no args
    try:
        results = await run_throttled(
            factory, todo, max_concurrency=num_workers, on_progress=on_progress
        )
    finally:
        if bar is not None:
            bar.close()

    written = 0
    dropped_incomplete = 0
    # Append as we go so a crash still leaves a resumable, valid JSONL.
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "a") as f:
        for pair, result in zip(todo, results):
            if isinstance(result, Exception):
                print(f"[nli-training] Generation failed for a pair: {result}")
                dropped_incomplete += 1
                continue
            samples = [str(mot) for mot in getattr(result, "sample_generations", [])]
            if len(samples) != expected_n:
                # Incomplete group (some generations failed / were unparsable):
                # cannot form fixed-width features, so drop it.
                dropped_incomplete += 1
                continue
            gold = pair["label"]
            labels = [1 if _sample_label(s) == gold else 0 for s in samples]
            rec = {
                "premise": pair["premise"],
                "hypothesis": pair["hypothesis"],
                "gold": gold,
                "samples": samples,
                "labels": labels,
            }
            f.write(json.dumps(rec) + "\n")
            written += 1

    summary = {
        "written": written,
        "skipped_existing": skipped_existing,
        "dropped_incomplete": dropped_incomplete,
    }
    print(f"[nli-training] Done: {summary}")
    return summary


def _read_groups(samples_path: str) -> tuple[list[list[str]], list[list[int]]]:
    """Read a samples JSONL into (training_samples, training_labels) groups."""
    training_samples: list[list[str]] = []
    training_labels: list[list[int]] = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            training_samples.append(list(rec["samples"]))
            training_labels.append([int(x) for x in rec["labels"]])
    return training_samples, training_labels


def _fit_classifier(
    strategy: SIMBAUQSamplingStrategy,
    training_samples: list[list[str]],
    training_labels: list[list[int]],
    *,
    progress: bool = False,
) -> ProbabilisticClassifier:
    """Fit a random forest on per-group similarity features.

    Mirrors :meth:`SIMBAUQSamplingStrategy._train_classifier` exactly (same
    ``_compute_similarity_matrix`` / ``_extract_features``), but iterates the
    groups here so an optional ``tqdm`` progress bar can wrap the compute-heavy
    similarity-matrix step.
    """
    try:
        from sklearn.ensemble import (
            RandomForestClassifier,  # type: ignore[import-not-found]
        )
    except ImportError:
        raise ImportError(
            "scikit-learn is required to train the classifier. "
            "It is a base dependency of fact_reasoner, so this "
            "usually means an incomplete install; reinstall or "
            "`pip install scikit-learn`."
        )

    groups = list(zip(training_samples, training_labels))
    if progress:
        from tqdm import tqdm

        groups = tqdm(groups, desc="Extracting features", unit="group")

    x_train: list[np.ndarray] = []
    y_train: list[int] = []
    for samples, labels in groups:
        sim_matrix = strategy._compute_similarity_matrix(samples)
        for i, label in enumerate(labels):
            x_train.append(strategy._extract_features(sim_matrix, i))
            y_train.append(label)

    clf = RandomForestClassifier(
        max_depth=strategy.clf_max_depth, random_state=strategy.clf_random_state
    )
    clf.fit(x_train, y_train)
    return clf


def train_classifier_from_jsonl(
    samples_path: str,
    *,
    temperatures: list[float] | None = None,
    n_per_temp: int = 4,
    similarity_metric: str = "rouge",
    clf_max_depth: int = 4,
    clf_random_state: int | None = 0,
    progress: bool = False,
) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Train a SIMBA-UQ classifier from a generated samples JSONL.

    Uses the same feature extraction the strategy applies at inference time
    (:meth:`SIMBAUQSamplingStrategy._compute_similarity_matrix` /
    ``_extract_features``), so the classifier sees identical features.

    Args:
        samples_path: JSONL produced by :func:`generate_training_samples`.
        temperatures, n_per_temp: Must match the values used at generation (they
            fix the group size / feature dimension).
        similarity_metric: Similarity metric to compute features with. Should
            match what will be used at inference.
        clf_max_depth, clf_random_state: Random-forest hyper-parameters.
        progress: If True, show a ``tqdm`` bar over per-group feature extraction
            (the compute-heavy part, especially for the ``sbert`` metric).

    Returns:
        A tuple ``(classifier, metadata)`` where ``metadata`` records the config
        needed to validate and reload the classifier.

    Raises:
        ValueError: If the JSONL is empty or contains a group whose size does not
            equal ``len(temperatures) * n_per_temp``.
    """
    training_samples, training_labels = _read_groups(samples_path)
    if not training_samples:
        raise ValueError(f"No training groups found in {samples_path!r}.")

    strategy = SIMBAUQSamplingStrategy(
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric=similarity_metric,
        confidence_method="aggregation",  # avoid requiring a classifier to construct
        clf_max_depth=clf_max_depth,
        clf_random_state=clf_random_state,
    )
    expected_n = len(strategy.temperatures) * strategy.n_per_temp
    for i, group in enumerate(training_samples):
        if len(group) != expected_n or len(training_labels[i]) != expected_n:
            raise ValueError(
                f"Group {i} in {samples_path!r} has size "
                f"{len(group)}/{len(training_labels[i])}, expected {expected_n} "
                "(len(temperatures) * n_per_temp). Regenerate with matching config."
            )

    clf = _fit_classifier(
        strategy, training_samples, training_labels, progress=progress
    )
    metadata = {
        "temperatures": list(strategy.temperatures),
        "n_per_temp": strategy.n_per_temp,
        "similarity_metric": similarity_metric,
        "n_features_in": expected_n - 1,
        "n_groups": len(training_samples),
    }
    return clf, metadata


def save_classifier(
    clf: ProbabilisticClassifier, path: str, metadata: dict[str, Any]
) -> None:
    """Persist a trained classifier together with its config metadata.

    Uses ``joblib`` (a base dependency).
    """
    try:
        import joblib  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "joblib is required to save the classifier. "
            "It is a base dependency of fact_reasoner, so this "
            "usually means an incomplete install; reinstall or "
            "`pip install joblib`."
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump({"clf": clf, "metadata": dict(metadata)}, path)
    print(f"[nli-training] Saved classifier to {path} (metadata: {metadata})")


def load_classifier(path: str) -> tuple[ProbabilisticClassifier, dict[str, Any]]:
    """Load a classifier and its metadata saved by :func:`save_classifier`.

    Returns:
        ``(classifier, metadata)``.

    Raises:
        ImportError: If ``joblib`` is unavailable.
        ValueError: If the file does not contain the expected structure.
    """
    try:
        import joblib  # type: ignore[import-not-found]
    except ImportError:
        raise ImportError(
            "joblib is required to load the classifier. "
            "It is a base dependency of fact_reasoner, so this "
            "usually means an incomplete install; reinstall or "
            "`pip install joblib`."
        )
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "clf" not in obj:
        raise ValueError(
            f"{path!r} is not a valid saved classifier "
            "(expected a dict with a 'clf' key from save_classifier)."
        )
    return obj["clf"], obj.get("metadata", {})


def evaluate_classifier(
    classifier: ProbabilisticClassifier,
    samples_path: str,
    *,
    temperatures: list[float] | None = None,
    n_per_temp: int = 4,
    similarity_metric: str = "rouge",
    progress: bool = False,
) -> dict[str, float]:
    """Compare classifier vs. aggregation *sample-selection* accuracy on held-out data.

    For each group, both methods pick the highest-confidence sample; the pick is
    "correct" if that sample's stored label is ``1``. Reports the fraction of
    groups where each method selected a correct sample (higher is better), plus the
    classifier's per-sample ``P(correct)`` AUROC-free hit rate.

    Args:
        classifier: A fitted classifier (from :func:`load_classifier` or
            :func:`train_classifier_from_jsonl`).
        samples_path: A held-out samples JSONL (e.g. generated from
            ``val_balanced.json``).
        temperatures, n_per_temp, similarity_metric: Must match how features were
            computed at training time.
        progress: If True, show a ``tqdm`` bar over the per-group evaluation.

    Returns:
        ``{"classifier_selection_acc", "aggregation_selection_acc", "n_groups"}``.
    """
    import numpy as np

    strategy = SIMBAUQSamplingStrategy(
        temperatures=temperatures,
        n_per_temp=n_per_temp,
        similarity_metric=similarity_metric,
        confidence_method="aggregation",
    )
    training_samples, training_labels = _read_groups(samples_path)

    groups = list(zip(training_samples, training_labels))
    if progress:
        from tqdm import tqdm

        groups = tqdm(groups, desc="Evaluating", unit="group")

    clf_hits = 0
    agg_hits = 0
    n = 0
    for samples, labels in groups:
        if len(samples) < 2:
            continue
        sim = strategy._compute_similarity_matrix(samples)
        # Classifier selection.
        feats = [strategy._extract_features(sim, i) for i in range(len(samples))]
        clf_conf = classifier.predict_proba(feats)[:, 1]
        clf_pick = int(np.argmax(clf_conf))
        clf_hits += int(labels[clf_pick] == 1)
        # Aggregation selection (data-free baseline).
        agg_conf = strategy._compute_confidences(sim)
        agg_pick = int(np.argmax(agg_conf))
        agg_hits += int(labels[agg_pick] == 1)
        n += 1

    result = {
        "classifier_selection_acc": clf_hits / n if n else 0.0,
        "aggregation_selection_acc": agg_hits / n if n else 0.0,
        "n_groups": float(n),
    }
    print(f"[nli-training] Evaluation: {result}")
    return result
