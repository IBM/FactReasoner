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

# LLM prompts for atom-atom relation mining (deep-dive Sections 4.2-4.3).
#
# All prompts are RESPONSE-GROUNDED: the FULL response is injected as context so
# the model asserts only relations the response actually draws. Judging an atom
# pair in ISOLATION makes the model accept any *abstractly plausible* relation,
# which over-connects the graph (the robust empirical failure mode: ~6-9
# relations/atom, spurious contradictions on a coherent paragraph). Grounding is
# mandatory -- there is no ungrounded/pair-only path.
#
# Two prompts implement the type-posterior x conditional-strength decomposition
# p = P(tau | a_i, a_j) x P(a_j | a_i, tau):
#
#   * Prompt A (PROMPT_SENSE_COUPLING) -- a chain-of-thought call over an ordered
#     atom pair (A, B), given the response, that names the Level-2 discourse SENSE
#     and maps it to a Level-1 COUPLING (one of the five: entailment,
#     contradiction, equivalence, exclusive, co_necessity -- see the revised
#     coherence_mrf_deepdive). The final answer is a bracketed [coupling=...] tag
#     whose token logprobs give the type confidence P(tau | a_i, a_j). The chain of
#     thought comes first so the model commits to the sense only after reasoning,
#     and only when the response draws the link.
#
#   * Prompt B -- given the coupling from Prompt A and the response, elicits the
#     conditional strength P(a_j | a_i, tau). TWO forms are provided:
#       - PROMPT_STRENGTH_SURROGATE (DEFAULT): a Yes/No surrogate-token question.
#         The strength is read as the renormalized token probability
#         p = P("Yes") / (P("Yes") + P("No")) from the answer token's logprobs, or
#         as the affirm-fraction over N samples when logprobs are unavailable. This
#         replaces the poorly-calibrated verbalized number with a quantity taken
#         from the model's own distribution (Kadavath et al. arXiv:2207.05221;
#         cf. EPK arXiv:2505.15918 for graphical-model parameters).
#       - PROMPT_STRENGTH (baseline): the older verbalized probability [p=0.NN],
#         kept only for comparison; verbalized confidence is known to be weakly
#         calibrated (Xiong et al. ICLR 2024, arXiv:2306.13063).
#
# All prompts mirror the style of ``core/nli.py`` (instruction + few-shots). The
# verbalized bracket span is kept as its own token run so the label and its
# probability are read from the SAME span (the EOS-drop / fused-bracket pitfalls
# documented in project memory).

# The set of Level-2 senses offered to the model, kept in sync with
# ``taxonomy.Level2Sense`` and interpolated into the prompt.
_SENSE_MENU = (
    "Cause-Effect, Effect-Cause, Evidence, Condition, Restatement, "
    "Instantiation, Contrast, Concession, Alternative, Disjunction, "
    "Precedence, Succession, None"
)


# ----------------------------------------------------------------------------
# Prompt A -- joint discourse sense + Level-1 coupling classification, grounded
# in the full response so the model asserts only relations the response draws.
# ----------------------------------------------------------------------------

PROMPT_SENSE_COUPLING = """

Instructions:
You are given the full RESPONSE a model produced, and two atomic claims, A and \
B, both taken FROM THAT RESPONSE, in their order of appearance (A comes before \
B). Your task is to decide the discourse/logical relation FROM A TO B, following \
the steps below.

IMPORTANT -- ground your decision in the response. Assert a coupling ONLY if the \
response ITSELF draws that connection between A and B (as written, or as a clear \
step in the author's argument/narrative). Do NOT assert a relation that is merely \
plausible in general but that the response does not actually make. If A and B \
both appear in the response yet the response draws no logical or discourse \
dependence between them, the answer is None.

1. Reason step by step, referring to the response: does the response present A \
as causing, enabling, providing evidence for, restating, elaborating, temporally \
preceding, contrasting with, or contradicting B? Is B a claim the response later \
withdraws or that a holding resolves? Consider the direction (A to B). If the \
response links A and B only indirectly through other claims, or not at all, that \
is None.

2. Name the DISCOURSE SENSE, one of: {{sense_menu}}.
   - Cause-Effect: A causes/leads to B. Effect-Cause: A is the effect, B its cause.
   - Evidence: A provides evidence for B. Condition: A is a condition for B.
   - Restatement: A and B assert the same thing. Instantiation: A is a general \
claim, B a specific instance (or vice versa).
   - Contrast: A and B are in opposition but NOT exhaustive (they need not cover \
all possibilities; both could conceivably be false).
   - Concession: A and B are in tension but the text concedes/resolves it \
("although A, still B", or a holding settles it).
   - Alternative: A and B are EXHAUSTIVE competing options -- EXACTLY ONE holds \
(they are mutually exclusive AND together cover the possibilities: not both, and \
not neither). E.g. "no one was harmed" vs "three people died"; "the cause was \
pilot error" vs "the cause was a metallurgical defect".
   - Disjunction: AT LEAST ONE of A and B holds (they may both hold, but the \
response rules out neither being true) -- e.g. two supporting findings at least \
one of which must be present.
   - Precedence/Succession: A and B are ordered in time with no truth dependence.
   - None: the response draws no logical or discourse dependence between A and B.

3. Map the sense to a COUPLING, one of: entailment, contradiction, \
equivalence, exclusive, co_necessity, none.
   - Cause-Effect, Effect-Cause, Evidence, Condition, Instantiation -> entailment
   - Restatement -> equivalence
   - Contrast, Concession -> contradiction
   - Alternative -> exclusive       (exactly one of A, B is true)
   - Disjunction -> co_necessity     (at least one of A, B is true)
   - Precedence, Succession, None -> none
   Prefer "exclusive" over "contradiction" when the two claims are not just \
incompatible but EXHAUSTIVE (one of them must be true); prefer "contradiction" \
when they merely cannot both hold but could both be false.

4. Give your final answer as two bracketed tags on ONE line, sense first:
[sense=Cause-Effect] [coupling=entailment]
A JSON object {"sense":"Cause-Effect","coupling":"entailment"} is also acceptable.

Use the following examples to better understand your task.

Example 1 (the response makes the causal link):
RESPONSE: The company launched a flawed product last quarter. Reviewers panned \
it, returns spiked, and the company's stock price fell 15 percent over the same \
period.
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
1. Reasoning: the response presents the flawed launch as the head of a chain \
(panning, returns) that ends in the stock decline, so the response itself draws a \
causal link from A to B.
2. Discourse sense: Cause-Effect.
3. Coupling: A causing B is a positive inferential link, i.e. entailment.
4. Final answer:
[sense=Cause-Effect] [coupling=entailment]

Example 2 (both claims present, but the response draws NO connection -> None):
RESPONSE: The quarterly report was published in April. Separately, the annual \
audit was scheduled for December. The two processes are run by different teams \
and were not related this year.
A: The quarterly report was published in April.
B: The annual audit was scheduled for December.
1. Reasoning: both claims appear in the response, and one might imagine a \
reporting-to-audit link in general, but this response explicitly treats them as \
separate and unrelated. The response draws no dependence from A to B.
2. Discourse sense: None.
3. Coupling: no dependence the response asserts, i.e. none.
4. Final answer:
[sense=None] [coupling=none]

Example 3 (the response states an EXHAUSTIVE alternative -> exclusive):
RESPONSE: The official statement said no one was harmed in the incident. However, \
the coroner's report confirmed that three people died in the incident.
A: No one was harmed in the incident.
B: Three people died in the incident.
1. Reasoning: the response sets A and B against each other ("However, ...") and \
they cannot both be true; but they also cannot both be false -- either people \
were harmed or they were not -- so exactly one holds. This is exhaustive, not a \
mere contrast. No holding resolves it.
2. Discourse sense: Alternative.
3. Coupling: exactly one of A, B is true, i.e. exclusive.
4. Final answer:
[sense=Alternative] [coupling=exclusive]

Example 4 (at least one must hold -> co_necessity):
RESPONSE: The defect was caught in review: at least one of the two independent \
checks -- the vibration analysis or the metallurgical assay -- flagged it.
A: The vibration analysis flagged the defect.
B: The metallurgical assay flagged the defect.
1. Reasoning: the response asserts the defect WAS caught by at least one check, so \
A and B cannot both be false; but both could hold (both checks may have flagged \
it). This is a disjunction, not an exclusion.
2. Discourse sense: Disjunction.
3. Coupling: at least one of A, B is true, i.e. co_necessity.
4. Final answer:
[sense=Disjunction] [coupling=co_necessity]

Your task:
RESPONSE: {{response}}
A: {{atom_a}}
B: {{atom_b}}
"""


# ----------------------------------------------------------------------------
# Prompt B (default) -- conditional strength via a Yes/No surrogate token,
# grounded in the response so the strength reflects how strongly the RESPONSE
# ties B to A (not an abstract judgment).
# ----------------------------------------------------------------------------
#
# The answer's FIRST WORD must be Yes or No, so its token logprobs give the
# renormalized surrogate probability p = P("Yes") / (P("Yes") + P("No")). "Yes"
# always means "the coupling's asserted implication is credible", so p is the
# strength of the coupling regardless of type. The judgment is GRADED / plausibility
# based -- "Yes" covers weak-but-real links too, not only near-certain ones -- so a
# merely plausible entailment does not read as a flat "No"; the graded confidence
# instead comes out in the renormalized logprob p (and, for sampling, the affirm
# fraction). For a contradiction we ask whether B is likely FALSE given A, so "Yes"
# still means the contradiction is credible. A relation the response only weakly
# supports gets a lower renormalized p even when it is abstractly plausible.

PROMPT_STRENGTH_SURROGATE = """

Instructions:
You are given the full RESPONSE, two claims A and B drawn from it, and a \
{{coupling}} relation that holds from A to B. Assuming A is TRUE, judge -- IN THE \
CONTEXT OF THIS RESPONSE -- whether the relation's implication about B is \
credible: at least plausible / more likely than not, NOT whether it is certain.

- entailment or equivalence: given A and how the response uses it, is B at least \
plausibly TRUE (more likely than not)?
- contradiction or exclusive: given A and how the response uses it, is B at least \
plausibly FALSE (more likely than not)? (For "exclusive", A and B are exhaustive \
alternatives, so A being true makes B false.)
- co_necessity: A and B are a pair of which at least one holds. Given the response \
rules out "neither", is it at least plausible that B holds when A does NOT?

Answer with a SINGLE WORD, the very first word of your reply: Yes or No.
- Answer "Yes" if the implication is credible/plausible (even if not certain).
- Answer "No" only if the implication is implausible or the response does not \
actually support it.
Do not output anything before the word Yes or No.

Example 1 (response supports a near-certain entailment):
RESPONSE: The new alloy is chemically identical to the certified reference \
alloy, so it meets the certified reference specification.
A: The new alloy is chemically identical to the certified reference alloy.
B: The new alloy meets the certified reference specification.
coupling: entailment
Answer: Yes

Example 2 (weak but plausible, and the response draws the link -- still Yes):
RESPONSE: The company launched a flawed product last quarter, and its stock \
price fell 15 percent over the same period.
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
coupling: entailment
Answer: Yes

Example 3 (clear contradiction stated by the response):
RESPONSE: The official statement said no one was harmed, but three people died \
in the incident.
A: No one was harmed in the incident.
B: Three people died in the incident.
coupling: contradiction
Answer: Yes

Your task:
RESPONSE: {{response}}
A: {{atom_a}}
B: {{atom_b}}
coupling: {{coupling}}
Answer: """


# ----------------------------------------------------------------------------
# Prompt B (baseline) -- verbalized conditional strength P(a_j | a_i, tau),
# grounded in the response.
# ----------------------------------------------------------------------------

PROMPT_STRENGTH = """

Instructions:
You are given the full RESPONSE and two claims A and B drawn from it. Assume \
claim A is TRUE and, IN THE CONTEXT OF THIS RESPONSE, estimate how strongly A \
determines B under a {{coupling}} relation from A to B.

- For an entailment or equivalence coupling: how likely is B to be TRUE given A?
- For a contradiction or exclusive coupling: how likely is B to be FALSE given A? \
(exclusive = A and B are exhaustive alternatives, so A true forces B false.)
- For a co_necessity coupling (at least one of A, B holds): how likely is B to be \
TRUE when A is FALSE?

Answer with a single probability in [0, 1] to two decimals, after one short \
sentence of justification. A near-certain link is close to 1.00; a merely \
plausible link is around 0.60-0.70. End your answer with the probability wrapped \
in brackets on its own, exactly in the form: [p=0.NN]

Example 1:
RESPONSE: The new alloy is chemically identical to the certified reference \
alloy, so it meets the certified reference specification.
A: The new alloy is chemically identical to the certified reference alloy.
B: The new alloy meets the certified reference specification.
coupling: entailment
Justification: chemical identity to a certified reference almost guarantees the \
specification is met, so B follows very strongly from A.
[p=0.95]

Example 2:
RESPONSE: The company launched a flawed product last quarter, and its stock \
price fell 15 percent over the same period.
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
coupling: entailment
Justification: a flawed product can plausibly drive a stock decline, but many \
other factors affect price, so the link is only moderate.
[p=0.65]

Example 3:
RESPONSE: The official statement said no one was harmed, but three people died \
in the incident.
A: No one was harmed in the incident.
B: Three people died in the incident.
coupling: contradiction
Justification: if no one was harmed, it is almost certain that B (three deaths) \
is false.
[p=0.93]

Your task:
RESPONSE: {{response}}
A: {{atom_a}}
B: {{atom_b}}
coupling: {{coupling}}
"""


def build_sense_coupling_prompt() -> str:
    """Return Prompt A (response-grounded) with the sense menu interpolated.

    The ``{{response}}`` / ``{{atom_a}}`` / ``{{atom_b}}`` placeholders remain for
    Mellea's ``user_variables`` substitution at call time.

    Returns:
        The Prompt A template string.
    """
    return PROMPT_SENSE_COUPLING.replace("{{sense_menu}}", _SENSE_MENU)


def build_surrogate_strength_prompt() -> str:
    """Return the default (surrogate Yes/No) conditional-strength prompt.

    The ``{{response}}`` / ``{{atom_a}}`` / ``{{atom_b}}`` / ``{{coupling}}``
    placeholders remain for Mellea's ``user_variables`` substitution at call time.
    The answer's first word is the surrogate token whose logprobs give the
    renormalized strength.

    Returns:
        The surrogate-token strength prompt template string.
    """
    return PROMPT_STRENGTH_SURROGATE


def build_strength_prompt() -> str:
    """Return the verbalized (baseline) conditional-strength prompt.

    The ``{{response}}`` / ``{{atom_a}}`` / ``{{atom_b}}`` / ``{{coupling}}``
    placeholders remain for Mellea's ``user_variables`` substitution at call time.

    Returns:
        The verbalized Prompt B template string.
    """
    return PROMPT_STRENGTH
