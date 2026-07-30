# NLI Probability via SIMBA-UQ with a Trained Classifier

Demonstrates the SIMBA-UQ **classifier** confidence method for NLI: instead of the data-free aggregation heuristic, a trained probabilistic classifier scores each generated sample's likelihood of being correct.

**Source:** [`docs/examples/core/ex_nli_simbauq_classifier.py`](examples/core/ex_nli_simbauq_classifier.py)

## Overview

SIMBA-UQ samples the NLI label several times across a range of temperatures and selects the most confident sample. There are two ways to score confidence:

- **`aggregation`** (default) — data-free: each sample is scored by how similar it is to the others (see [`ex_nli_simbauq`](ex_nli_simbauq.md)).
- **`classifier`** — a probabilistic classifier maps a sample's pairwise-similarity feature vector to `P(correct)`. This usually selects better samples, but it must be *trained* first on labeled NLI data.

This example loads a classifier trained by `scripts/train_simbauq_nli.py` and uses it at inference.

## Prerequisites

- No extra install: `scikit-learn` and `joblib` (training/loading the classifier) and `rouge-score` (the default metric) are base dependencies.
- A Mellea backend (`--backend {rits,ollama,vllm,openai}`, defaults to Ollama; `openai` covers hosted frontier models, including Claude via `--base-url https://api.anthropic.com/v1/`). See [`ex_nli_simbauq`](ex_nli_simbauq.md) for backend details.
- A **trained classifier**. Produce one from labeled `{premise, hypothesis, label}` NLI data:

  ```bash
  python scripts/train_simbauq_nli.py --stage all \
    --nli-data /path/to/train_balanced.json --num-pairs 900 \
    --backend ollama --similarity-metric rouge \
    --samples artifacts/simbauq_nli_samples.jsonl \
    --out artifacts/simbauq_nli_clf.joblib
  ```

  The trainer runs SIMBA-UQ generation over a balanced subset of the labeled pairs (each sample is labeled correct iff its extracted NLI label matches the gold label), then fits and saves the classifier. Generation is cached in the `--samples` JSONL and is resumable; scale `--num-pairs` to your compute budget.

## How It Works

1. Build a Mellea backend via `--backend`.
2. Construct an `NLIExtractor` with `nli_method="simbauq"`, `simbauq_confidence_method="classifier"`, and `simbauq_classifier_path=<saved .joblib>`.
3. The extractor loads the classifier and **validates its feature dimension** against `len(temperatures) * n_per_temp - 1`. A classifier trained under a different temperature schedule / `n_per_temp` therefore fails fast with a clear error.

   > ⚠️ That check is dimension-only, so it does **not** catch a mismatched
   > `--similarity-metric`: the feature vector is the same length whichever metric
   > produced it, so a mismatch silently yields meaningless confidences rather than an
   > error. Make sure the metric matches training yourself.
4. `nli.run(premise, hypothesis)` samples across temperatures, scores each sample with the classifier, and returns the winning sample's label and its `P(correct)` as the probability.

## Usage

```bash
python docs/examples/core/ex_nli_simbauq_classifier.py \
  --backend ollama \
  --classifier-path artifacts/simbauq_nli_clf.joblib \
  --similarity-metric rouge
```

The SIMBA-UQ preamble printed at startup confirms the classifier was loaded (`confidence method: classifier (loaded from ...)`).

## Wiring into a full assessment

The classifier is also reachable from the CLI, so a full FactReasoner run uses it for every NLI relation:

```bash
fact-reasoner --pipeline factreasoner --merlin-path <merlin> \
  --backend ollama --nli-method simbauq \
  --nli-confidence-method classifier \
  --nli-classifier-path artifacts/simbauq_nli_clf.joblib \
  --query "..." --response "..." --output-file result.json
```

## Output

The script prints the premise, the hypothesis, the predicted NLI label (`entailment` / `contradiction` / `neutral`), and its probability in `[0, 1]`.
