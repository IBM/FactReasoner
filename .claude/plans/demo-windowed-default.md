# Plan: windowed candidate pairs by default in the coherence analysis

## Today

The demo's `Settings.pair_policy` defaults to `"all_pairs"`
(`src/fact_reasoner/demo/engine.py:164`). The docstring above it explains why
(`engine.py:147-150`): "the demo inputs are short, so all pairs is affordable".
That is the one place where the demo departs from the CLI, whose `--pair-policy`
defaults to `windowed` (`lcs/cli.py:211`), as does `candidate_pairs.select`.

The front end reads its defaults from the server (`GET /api/config` returns
`defaults: Settings().__dict__`, `demo/app.py:187`). So the dropdown, the
*Default* preset (`values: {}`, i.e. the server defaults) and the "N advanced
settings differ" count all follow `Settings` without any change.

## Change

1. **`engine.py`**: `pair_policy: str = "windowed"`. The window radius keeps its
   default of 4, the same as the CLI. Rewrite the docstring: the defaults now
   follow the CLI with no exception. Windowed pairs are forward-only within the
   radius, refined by the response's discourse signals (promote / demote), so a
   run makes fewer miner calls and the graph is less over-connected.
2. **Presets (`static/app.js:149-160`)**: no code change needed.
   - *Fast* keeps `pair_policy: "windowed", window: 2`. It still differs from
     the default, but now only in its radius.
   - *Default* follows the new server default.
   - *Thorough* already sets `pair_policy: "all_pairs"` explicitly, so it still
     scores every pair. It is now the only preset that does.
   - Optional: change *Fast*'s note to "Fewer passages, a narrower pair window:
     quickest." The current note, "Fewer passages and claim pairs", is still
     true, so I would leave it unless you want the wording changed.
3. **Docs (`docs/examples/lcs/demo.md:304-308`)**: in the preset table, *Fast*
   becomes "windowed radius 2 (default radius 4)". *Thorough* gains "all claim
   pairs", since that is now a change from the defaults. If the settings table
   (`demo.md:325`) or anywhere else says the demo default is all pairs, update
   it to say windowed.
4. **Tests (`tests/test_demo.py`)**:
   - Add a check that `Settings().pair_policy == "windowed"` and that
     `/api/config` reports it in `defaults`.
   - `test_stub_run_on_free_text` sets `all_pairs` explicitly and stays as it
     is, so the all-pairs path keeps its coverage.
   - Check that the stub / dry-run tests that use the defaults still pass with
     windowed pairs. Windowed needs the response text, and `_miner` already
     passes it through. If a test asserts exact relation counts on the default
     settings, pin `pair_policy: "all_pairs"` in that test instead of changing
     the counts.

## Not affected

- **Offline replay of bundled examples:** the claims and relations come from the
  example's `gold` graph, so the pair policy plays no part.
- **Factuality's `nli_mode`:** a separate knob, still `fast`.
- **Request bodies that set `pair_policy` explicitly:** they behave as before.

## Verify

- `pytest tests/test_demo.py tests/test_lcs_relation_miner.py`
- Run the demo with `fact-reasoner-lcs-demo`, then check the following:
  - The *Candidate pairs* dropdown shows *windowed* on load.
  - *Default* is the active preset.
  - Picking *Thorough* switches the dropdown to *all_pairs*, and *Default*
    switches it back.
  - A coherence run on a free-text example completes, with a "pairs" progress
    count below n·(n−1)/2.
