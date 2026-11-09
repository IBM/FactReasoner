# Compare a skill's instructions and Python code

This experimental workflow asks two questions about one skill.

| Evidence | Claim being checked | What the result helps find |
| --- | --- | --- |
| Skill instructions | An observed code behavior | Behavior the instructions may leave out |
| Python source | A requirement in the instructions | Requirements with limited or opposing implementation evidence |

The comparison extracts requirements from Markdown and recognizes Python
operations and selected Mellea constructs from the syntax tree. It records
source lines and uses FactReasoner's NLI extractor and Merlin inference on
separate graphs for the two directions. The requirement search can also follow
statically resolved calls and combine connected statements, such as an
instruction call, its requirements list, and a validator.

## Try the example

Install FactReasoner from this checkout, then run the example from the
repository root. The synthetic skill documents an HTTP GET and cleanup.
Its Python source includes the GET and an additional subprocess call.

```sh
python -m fact_reasoner.skill_consistency \
  --specification docs/examples/skill_consistency/sample/SKILL.md \
  --code-root docs/examples/skill_consistency/sample \
  --output-dir /tmp/skill-consistency-plan \
  --prepare-only
```

Preparation runs locally without model calls or a Merlin binary.
`prepare.json` contains the extracted statements, source locations and hashes,
and eligible comparisons. The tool reads Python source without importing or
running the skill. It skips symlinked Python files, hidden paths, and directories
named `tests`, `fixtures`, `examples`, or `intermediate` within the selected code root.

To judge the pairs, use a new output directory, remove `--prepare-only`, and
add a model and the path to the [Merlin executable](https://github.com/radum2275/merlin).
For a locally served vLLM model:

```sh
python -m fact_reasoner.skill_consistency \
  --specification docs/examples/skill_consistency/sample/SKILL.md \
  --code-root docs/examples/skill_consistency/sample \
  --output-dir /tmp/skill-consistency-result \
  --backend vllm --model MODEL_NAME --base-url http://localhost:8000/v1 \
  --merlin-path /path/to/merlin
```

The model endpoint must return token log probabilities. The NLI model receives
normalized statements derived from the selected inputs. Use a model endpoint
approved for the inputs. Generated reports can contain source excerpts.

## Read the results

- `analysis.json` includes the two relation sets, behavior-level documentation
  support, and requirement coverage. It retains raw model labels, ambiguity,
  guard decisions, and native FactReasoner diagnostics.
- The two SVGs show support and contradiction edges. The JSON retains neutral
  comparisons and targets with no accepted comparison.
- `--mode flat` judges every eligible implementation-to-requirement pair.
  `--mode tree` can stop a branch once its requirement group is supported.
  `--mode tree-audit` visits all eligible pairs while preserving the traversal
  trace. These modes share the same documentation-to-code check.
- `--max-judgments` bounds the eligible pair universe before scoring. It does
  not silently remove pairs or cap backend retries.

Coverage measures the available static evidence. An unassessed requirement may
come from a missing candidate, an unsupported construct, or a withheld judgment.
The result cannot establish runtime behavior or the absence of undocumented code.
