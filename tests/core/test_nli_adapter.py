# coding=utf-8
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

"""Unit tests for the NLI LoRA adapter extractor (fact_reasoner.core.nli)."""

import asyncio
import json
import math
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mellea.backends import ModelOption

from fact_reasoner.backends import build_backend
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.nli import (
    INSTRUCTION_NLI_ADAPTER,
    NLIAdapterExtractor,
    NLIExtractor,
    nli_adapter_prompt,
)
from fact_reasoner.core.nli_cache import extractor_identity
from fact_reasoner.core.utils import build_relations

AACT = "fact_reasoner.core.nli.mfuncs.aact"

PREMISE = (
    "The author expresses frustration with loud advertisements on YouTube, which "
    "are significantly louder than the actual content being watched. These "
    "advertisements often have poor audio engineering, with voices, effects, and "
    "background music pushed to maximum volume. The author suggests that this "
    "approach is counterproductive, as it leads to viewers ignoring or blocking "
    "the ads."
)
HYPOTHESIS = (
    "The commercials themselves are not what are loud, the actual show or content "
    "that the viewer is watching/listening to is quiet."
)
# The user message of one training example, verbatim.
TRAINING_USER_MESSAGE = (
    f"Text A (premise): {PREMISE}\n\nText B (hypothesis): {HYPOTHESIS}\n\n"
    "<nli>As a judge agent, your role is to classify the logical relationship "
    "between the two provided texts according to the given criteria, utilizing all "
    "available information, including conversations, documents, and tools.\n\n### "
    "Criteria: Natural language inference judges the logical relationship from a "
    "premise (text A) to a hypothesis (text B), assuming both describe the same "
    "situation. The relationship is exactly one of three labels.\n\n'entailment': a "
    "reader accepting the premise as true must also accept the hypothesis as true. "
    "The hypothesis adds no unsupported information; it may restate the premise, "
    "paraphrase it, generalize it, or state part of it.\n\n'contradiction': a reader "
    "accepting the premise as true must reject the hypothesis as false. The premise "
    "and hypothesis cannot both hold of the same situation, because they conflict "
    "over an entity, action, attribute, quantity, time, place, or negation.\n\n"
    "'neutral': the premise neither compels nor rules out the hypothesis. The "
    "hypothesis could be true or false while the premise stays true — typically "
    "because it adds detail the premise never mentions, states a plausible but "
    "unstated motive, cause, or outcome, or makes a broader claim the premise only "
    "partly supports.\n\nApply these rules:\n- Judge only what the premise states or "
    "necessarily implies. Do not rely on outside world knowledge, and do not assume "
    "unstated details.\n- Unmentioned is not the same as false. Information absent "
    "from the premise makes a hypothesis 'neutral', not 'contradiction'.\n- "
    "Plausible is not the same as entailed. A likely inference the premise does not "
    "guarantee is 'neutral', not 'entailment'.\n- A hypothesis more specific than "
    "the premise is usually 'neutral'; a hypothesis more general than the premise "
    "is usually 'entailment'.\n- Weigh negation, quantifiers ('all', 'some', "
    "'none'), and numbers carefully: they often flip the label on their own.\n- "
    "Choose 'contradiction' only for genuine incompatibility, not mere difference "
    "in wording, emphasis, or level of detail.\n\n### Scoring Schema: Judge the "
    "relationship from text A to text B and return exactly one of 'entailment', "
    "'contradiction', or 'neutral'. Provide the final answer as a JSON dict with "
    'the following format: {"label": "entailment" | "contradiction" | "neutral"}.'
)


def _backend():
    backend = MagicMock()
    backend.model_id = "fr-nli"
    return backend


class TestPrompt:
    def test_matches_training_data_verbatim(self):
        assert nli_adapter_prompt(PREMISE, HYPOTHESIS) == TRAINING_USER_MESSAGE

    def test_instruction_is_the_shared_tail(self):
        assert TRAINING_USER_MESSAGE.endswith(INSTRUCTION_NLI_ADAPTER)


class TestRequest:
    def test_sends_one_plain_user_message(self):
        aact = AsyncMock(return_value=MagicMock(success=False))
        with patch(AACT, new=aact):
            NLIAdapterExtractor(_backend()).run(PREMISE, HYPOTHESIS)

        action = aact.call_args.args[0]
        assert action.role == "user"
        assert action.content == TRAINING_USER_MESSAGE
        assert aact.call_args.kwargs["return_sampling_results"] is True

    def test_logprobs_method_requests_logprobs_without_reasoning(self):
        options = NLIAdapterExtractor(_backend())._model_options()
        assert options == {
            "logprobs": True,
            "top_logprobs": 5,
            ModelOption.THINKING: False,
        }

    def test_simbauq_method_only_disables_reasoning(self):
        nli = NLIAdapterExtractor(_backend(), nli_method="simbauq")
        assert nli._model_options() == {ModelOption.THINKING: False}

    def test_reasoning_none_leaves_backend_default(self):
        nli = NLIAdapterExtractor(_backend(), nli_method="simbauq", reasoning=None)
        assert nli._model_options() is None

    def test_failed_sampling_falls_back_to_neutral(self):
        with patch(AACT, new=AsyncMock(return_value=MagicMock(success=False))):
            result = NLIAdapterExtractor(_backend()).run(PREMISE, HYPOTHESIS)
        assert result == {"label": "neutral", "probability": 1.0}

    def test_few_shot_extractor_is_unchanged(self):
        assert NLIExtractor.prompt_format == "few_shot"
        assert NLIExtractor(_backend())._model_options() == {
            "logprobs": True,
            "top_logprobs": 5,
        }


class TestCacheIdentity:
    def test_adapter_verdicts_never_share_few_shot_keys(self):
        few_shot = extractor_identity(NLIExtractor(_backend()))
        adapter = extractor_identity(NLIAdapterExtractor(_backend()))
        # Same served model id on purpose: the prompt format alone separates them.
        assert few_shot == ("fr-nli", "logprobs")
        assert adapter == ("fr-nli", "logprobs+adapter")


# Token stream of the adapter's verdict, with logprobs for the label tokens.
_TOKENS = [
    ('{"', -0.01),
    ("label", -0.01),
    ('":', -0.01),
    (' "', -0.01),
    ("contr", -0.2),
    ("adiction", -0.1),
    ('"}', -0.01),
]


class _FakeVLLM(BaseHTTPRequestHandler):
    """A minimal /v1/chat/completions endpoint answering like the adapter."""

    requests: list = []

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "".join(t for t, _ in _TOKENS)},
        }
        if body.get("logprobs"):
            choice["logprobs"] = {
                "content": [
                    {
                        "token": t,
                        "logprob": lp,
                        "bytes": None,
                        "top_logprobs": [{"token": t, "logprob": lp, "bytes": None}],
                    }
                    for t, lp in _TOKENS
                ]
            }
        data = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [choice],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 7,
                    "total_tokens": 8,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def fake_vllm():
    _FakeVLLM.requests = []
    server = HTTPServer(("127.0.0.1", 0), _FakeVLLM)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield build_backend(
            "vllm",
            model_id="fr-nli",
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
    finally:
        server.shutdown()


class TestWireFormatAndPipeline:
    """End to end through Mellea's vLLM (OpenAI-compatible) backend."""

    def test_logprobs_request_and_verdict(self, fake_vllm):
        nli = NLIAdapterExtractor(fake_vllm)
        (result,) = asyncio.run(nli.run_batch([PREMISE], [HYPOTHESIS]))

        (request,) = _FakeVLLM.requests
        assert request["model"] == "fr-nli"
        # Exactly the training-data input, nothing added (no template, no
        # rendered requirements).
        assert request["messages"] == [
            {"role": "user", "content": TRAINING_USER_MESSAGE}
        ]
        assert request["logprobs"] is True
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        # The label's own tokens set the probability, as for the few-shot path.
        assert result["label"] == "contradiction"
        assert result["probability"] == pytest.approx(math.exp((-0.2 - 0.1) / 2))

    def test_simbauq_samples_the_same_message(self, fake_vllm):
        nli = NLIAdapterExtractor(fake_vllm, nli_method="simbauq")
        (result,) = asyncio.run(nli.run_batch([PREMISE], [HYPOTHESIS]))

        assert len(_FakeVLLM.requests) == 20  # 4 temperatures x 5 samples
        assert all(
            r["messages"] == [{"role": "user", "content": TRAINING_USER_MESSAGE}]
            for r in _FakeVLLM.requests
        )
        assert {r["temperature"] for r in _FakeVLLM.requests} == {0.3, 0.5, 0.7, 1.0}
        assert result["label"] == "contradiction"
        assert 0.0 <= result["probability"] <= 1.0

    def test_verdict_becomes_a_pipeline_relation(self, fake_vllm):
        atom = Atom(id="a0", text=HYPOTHESIS)
        context = Context(id="c_a0_0", atom=atom, text=PREMISE)
        atom.add_contexts([context])

        relations = build_relations(
            atoms={"a0": atom},
            contexts={"c_a0_0": context},
            nli_extractor=NLIAdapterExtractor(fake_vllm),
            rel_atom_context=True,
            rel_context_context=False,
        )

        (relation,) = relations
        assert (relation.source.id, relation.target.id) == ("c_a0_0", "a0")
        assert relation.type == "contradiction"
        assert relation.link == "context_atom"
        assert relation.probability == pytest.approx(math.exp((-0.2 - 0.1) / 2))
