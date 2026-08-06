# `locobench_claude.json` -- Phase 2 generation config

Claude authors every item; a cross-vendor frontier committee validates it.

Phase 2 generation: Claude authors every item, a cross-vendor frontier committee
validates. Every entry is on the IBM LiteLLM gateway, reached through the 'openai'
backend kind -- the base_url selects the provider, not the kind -- so one
OPENAI_API_KEY (set to the gateway token) covers generator and committee alike.
The committee is 3 models spanning 3 real vendors. `family` is set EXPLICITLY on every
entry because it defaults to name.split('-')[0], which is how an earlier all-Claude
committee labelled a-/b-/c-/d- passed the '>= 3 distinct families' check spuriously.
Two GPT models plus one Gemini would be only 2 genuine families and is refused.
`auditor` is deliberately absent: naming one collapses the panel to that model alone,
and the panel is what gives V1 its any-of rule and V3/V4 their majority.

## Running it

```sh
export OPENAI_API_KEY="$ANTHROPIC_AUTH_TOKEN"   # the IBM LiteLLM gateway token
locobench-generate --config configs/locobench_claude.json --dry-run   # offline smoke
locobench-generate --config configs/locobench_claude.json            # 2 families = 10 items
```

Check the printed `[locobench] validation panel for ...` line reports **3 raters** before
trusting a run. `_build_auditors` degrades with a warning rather than aborting, so a model
that fails to build would otherwise silently shrink the panel -- and at one rater, the
majority rules that V3 and V4 depend on stop protecting anything.

Note the config carries no `_comment` key: `GenConfig.from_dict` rejects unknown keys as
typos, which is why this rationale lives here instead.
