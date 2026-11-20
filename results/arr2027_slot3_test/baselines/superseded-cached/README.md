GPT-6 judge results superseded on 2026-11-19: the five runs per response were byte-identical
requests, and the LiteLLM gateway's response cache served all five from the first completion
(every row has sd = 0). Re-run with LOCOBENCH_NO_CACHE=1, which makes the judge send
`extra_body={"cache": {"no-cache": true}}`. Kept for the record; not read by the evaluation.
