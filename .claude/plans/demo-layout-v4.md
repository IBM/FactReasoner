# Plan: demo revision (settings always open, analysis choice under the response)

## 1. Settings panel: always open, two aligned sub-panels
- Remove the Hide / Show settings button, the collapsed summary line, and the
  auto-collapse when a run starts. The panel stays open.
- Keep the two cards side by side: Factuality pipeline (blue), Coherence pipeline (purple).
- Lay out both cards on the same fixed grid so fields line up:
  - every field = label row (one line, fixed height) + one control row (32px);
  - the model field spans two columns in both cards, so the RITS|Ollama switch and the
    model dropdown start at the same x in both cards;
  - fixed-value boxes (`kev` NLI, `verbalized` strength) get the control height so they
    line up with the dropdowns next to them;
  - helper text goes under the label, never inside the control, so it can't push rows.
- Grid per card (3 columns, equal widths):

  ```
  Factuality                                  Coherence
  [Model: RITS|Ollama  ][model ▾   ]          [Relation miner: RITS|Ollama][model ▾ ]
  [NLI       ][Evidence     ][FactReasoner]   [Strength  ][Prompt      ][Sense menu ]
  [Per claim ][Pairs        ][            ]   [Pairs     ][Window      ][Reconcile  ]
  [☐ Decontextualize claims ][☐ Summarize]   [i-bound   ][Concession  ][☐ Evidence span]
                                              [LCS readouts ☑☑☐☑ (one row)        ]
  ```
- Remove "More coherence settings" / "Fewer coherence settings". All coherence
  settings are shown, in the grid above.
- Claim priors leave the settings panel, because the analysis choice below decides them.

## 2. Analysis choice under the Response box (replaces the header switch)
- Remove Factuality | Coherence | Both from the header.
- Under the response and evidence editors, above the Analyze button, add one Carbon
  selection of three options, as a segmented content switcher:

  `[ Factuality ]  [ Coherence ]  [ Coherence with factuality priors ]`

  | Option | Runs | Claim priors | Results panel shows |
  |---|---|---|---|
  | Factuality (default) | claims → evidence → NLI → inference | n/a | Factuality only |
  | Coherence | claims → relation miner → MRF | uniform 0.5 | Coherence only |
  | Coherence with factuality priors | both, factuality posteriors → MRF priors | factuality posteriors | Factuality and Coherence tabs |

- The evidence editor (pasted evidence) shows only for options that run factuality.
- The Analyze button label follows the choice ("Check factuality" / "Analyze coherence" /
  "Analyze coherence with factuality priors").
- Settings cards dim (not hide) when their pipeline is not part of the chosen analysis,
  with a note such as "not used by this analysis". The panel stays visible and aligned.

## 3. Results panel follows the choice
- Factuality: only the Factuality view; the tab strip is hidden.
- Coherence: only the Coherence view; tab strip hidden.
- Coherence with factuality priors: both tabs, opening on Coherence (the end result), with
  Factuality one click away.
- Claim rows keep the matching bars: factuality only, coherence (prior tick at 0.5), or both.

## 4. Engine / API
- Keep `mode` = `factuality | coherence | both` internally (`both` = coherence with
  factuality priors), so the API and tests barely change.
- Coherence mode always uses uniform priors (0.5). This drops the `flat` π slider and
  the `manual` per-claim prior sliders, plus the instant no-model-call re-score that
  depends on them. See open question.
- Default mode becomes `factuality`.

## 5. Tests and checks
- Update tests: default mode, coherence mode always uniform, examples still load.
- Screenshots: each of the 3 options, light and dark, phone width. Check field
  alignment by measuring element positions in both cards, not just by eye.

## Open question
The manual per-claim prior sliders (drag a prior and coherence re-scores with no model
calls) and the flat π slider don't fit "coherence only = uniform priors". Remove them,
or keep the per-claim sliders as an optional "what-if" after a coherence run?
