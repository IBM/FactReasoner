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

# The restricted menu: only the senses the LoCoBench corpora actually label.
# `Condition` and `Instantiation` are dropped -- both compile to entailment, both
# are semantically broad enough that a model reaching for "related somehow" lands
# on them, and both appear ZERO times in gold while accounting for 13% of one
# measured arm's edges. Kept in sync with `taxonomy.GOLD9_SENSES`.
_SENSE_MENU_GOLD9 = (
    "Cause-Effect, Effect-Cause, Evidence, Restatement, "
    "Contrast, Concession, Alternative, Disjunction, Precedence, None"
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


# ----------------------------------------------------------------------------
# Prompt A v2 -- same task, rebalanced for precision.
#
# v1 over-asserts: measured against a corpus where 7.2% of atom pairs are
# related, it answers "related" on 72% of the pairs a windowed policy hands it.
# Five properties of v1 push that way, and v2 addresses each:
#
#   1. Few-shot label balance was 3 related : 1 unrelated. v2 is 2 : 3. Few-shot
#      class balance is a strong prior on the answer distribution, and v1's was
#      pointed the wrong way by an order of magnitude relative to the truth.
#   2. v1's single negative had the response say the two claims "were not related
#      this year" -- an EXPLICIT disavowal. That teaches that None requires denial,
#      whereas real prose simply stays silent. v2's negatives are all silence:
#      topically adjacent claims the text never links.
#   3. v1 asked the model to be grounded but gave it nowhere to show the grounding,
#      and nothing checked it. v2 requires a verbatim [evidence=...] span QUOTED
#      FROM THE RESPONSE, which the parser can verify -- the only precision
#      mechanism here that is checkable rather than merely requested. (It must
#      quote the response, not the atoms: atoms are decontextualized rewrites and
#      only ~27% of them appear in the response even after normalization, whereas
#      the response is literal source text.)
#   4. v1's step 1 was a leading question listing eight affirmative relations
#      before mentioning None. v2 asks the discriminating question first: does the
#      text draw the link, or are the claims merely near each other?
#   5. v1 gave None one line against nine for relation-bearing senses. v2 gives it
#      first position and equal prose.
#
# Also named explicitly: transitivity. ~85% of v1's false positives are the model
# closing a chain the text laid out pairwise (96.6% of them touch a gold-edge
# endpoint), so v2 forbids inferring A-C from stated A-B and B-C.
# ----------------------------------------------------------------------------

PROMPT_SENSE_COUPLING_V2 = """

Instructions:
You are given the full RESPONSE a model produced, and two atomic claims, A and B, \
both taken FROM THAT RESPONSE. Decide the discourse/logical relation FROM A TO B.

Most pairs of claims in a text are NOT related. They are simply near each other. \
Your default answer is None, and you should depart from it only when the response \
itself draws a dependence between A and B.

Three things that are NOT relations, however plausible they look:
  - Topical adjacency. A and B being about the same subject, or sitting in \
neighbouring sentences, is not a relation.
  - World knowledge. If the link is true in general but the response does not \
draw it, the answer is None.
  - Transitivity. If the response links A to some third claim C, and C to B, that \
does NOT make A related to B. Report only the links the text draws directly.

1. First ask the discriminating question: does the response ITSELF assert a \
dependence from A to B -- as written, or as an explicit step in the author's \
argument? Identify the words in the response that do it. If you cannot point to \
such words, the answer is None.

2. If and only if you can, name the DISCOURSE SENSE, one of: {{sense_menu}}.
   - None: the response draws no dependence between A and B. This is the common \
case: unrelated, merely adjacent, related only by world knowledge, or related \
only through a third claim.
   - Cause-Effect: the response says A causes/leads to B. Effect-Cause: A is the \
effect, B its cause.
   - Evidence: the response offers A as evidence/support for B.
   - Restatement: A and B say the same thing -- a paraphrase, a converse, or a \
restatement the text marks ("in other words", "equivalently", "that is").
   - Contrast: A and B are opposed but NOT exhaustive (both could be false).
   - Concession: A and B are in tension and the text concedes or resolves it \
("although A, still B"; a holding settles it).
   - Alternative: A and B are EXHAUSTIVE competing options -- exactly one holds \
(not both, not neither).
   - Disjunction: AT LEAST ONE of A and B holds; both may.
   - Precedence: A and B are ordered in time with no truth dependence.

3. Map the sense to a COUPLING, one of: entailment, contradiction, equivalence, \
exclusive, co_necessity, none.
   - Cause-Effect, Effect-Cause, Evidence -> entailment
   - Restatement -> equivalence
   - Contrast, Concession -> contradiction
   - Alternative -> exclusive       (exactly one of A, B is true)
   - Disjunction -> co_necessity    (at least one of A, B is true)
   - Precedence, None -> none
   Prefer "exclusive" when the two claims are incompatible AND exhaustive; \
"contradiction" when they merely cannot both hold but could both be false.

4. Answer on ONE line, with three bracketed tags. The evidence must be a SHORT \
VERBATIM QUOTE COPIED FROM THE RESPONSE -- the words that draw the link. Do not \
quote claim A or claim B; quote the response. Use [evidence=none] when the sense \
is None.
[sense=Evidence] [coupling=entailment] [evidence="which is why the panel held"]

Use the following examples to better understand your task.

Example 1 (adjacent and on-topic, but the response draws no link -> None):
RESPONSE: The survey was administered in March to a sample of 400 households. \
Response rates in rural districts have declined over the past decade. The \
analysis weighted results by district population.
A: The survey was administered in March to a sample of 400 households.
B: Response rates in rural districts have declined over the past decade.
1. The two sentences are adjacent and both concern the survey, but the response \
never says the March administration bears on rural response rates. There are no \
words that draw a dependence.
2. Discourse sense: None.
3. Coupling: none.
4. [sense=None] [coupling=none] [evidence=none]

Example 2 (the response makes the causal link explicitly):
RESPONSE: The company launched a flawed product last quarter. Reviewers panned \
it, returns spiked, and as a direct result the company's stock price fell 15 \
percent over the same period.
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
1. The response presents the launch as the head of a chain ending in the stock \
decline, and marks it: "as a direct result".
2. Discourse sense: Cause-Effect.
3. Coupling: entailment.
4. [sense=Cause-Effect] [coupling=entailment] [evidence="as a direct result"]

Example 3 (true in general, but this response does not draw it -> None):
RESPONSE: Higher interest rates raise the cost of corporate borrowing. The firm \
opened three distribution centres last year. Its logistics costs fell by eight \
percent.
A: Higher interest rates raise the cost of corporate borrowing.
B: The firm opened three distribution centres last year.
1. One could argue borrowing costs affect expansion decisions, but that is world \
knowledge. This response states the two facts side by side and links neither to \
the other.
2. Discourse sense: None.
3. Coupling: none.
4. [sense=None] [coupling=none] [evidence=none]

Example 4 (the response states an exhaustive alternative -> exclusive):
RESPONSE: The two readings are mutually exclusive and exactly one must be right: \
either no one was harmed in the incident, or three people died in it.
A: No one was harmed in the incident.
B: Three people died in the incident.
1. The response sets A and B against each other and says exactly one must be \
right: "mutually exclusive and exactly one must be right: either ... or".
2. Discourse sense: Alternative.
3. Coupling: exclusive.
4. [sense=Alternative] [coupling=exclusive] [evidence="mutually exclusive and \
exactly one must be right"]

Example 5 (linked only through a third claim -> None):
RESPONSE: The alloy is chemically identical to the certified reference. It \
therefore meets the reference specification. Meeting that specification is a \
precondition for airframe use.
A: The alloy is chemically identical to the certified reference.
B: Meeting that specification is a precondition for airframe use.
1. The response links A to the middle claim (it meets the specification), and the \
middle claim to B. It does not link A to B directly; that would be transitivity, \
which is not a relation to report.
2. Discourse sense: None.
3. Coupling: none.
4. [sense=None] [coupling=none] [evidence=none]

Your task:
RESPONSE: {{response}}
A: {{atom_a}}
B: {{atom_b}}
"""

# Prompt A variants selectable by the miner. "v1" is byte-identical to the
# original and is kept permanently so any prompt claim can be A/B'd rather than
# asserted.
PROMPT_VARIANTS = ("v1", "v2")


def build_sense_coupling_prompt(
    variant: str = "v1", menu: str = "full"
) -> str:
    """Return Prompt A (response-grounded) with the sense menu interpolated.

    Args:
        variant: ``"v1"`` (the original) or ``"v2"`` (rebalanced for precision;
            requires a verifiable evidence span -- see the module comment above
            :data:`PROMPT_SENSE_COUPLING_V2`).
        menu: ``"full"`` for the whole sense taxonomy, or ``"gold9"`` for only the
            senses the LoCoBench corpora label.

    Returns:
        The Prompt A template string.

    Raises:
        ValueError: If `variant` or `menu` is unknown.
    """
    if variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt variant {variant!r} (expected one of "
            f"{list(PROMPT_VARIANTS)})."
        )
    if menu == "full":
        sense_menu = _SENSE_MENU
    elif menu == "gold9":
        sense_menu = _SENSE_MENU_GOLD9
    else:
        raise ValueError(f"Unknown sense menu {menu!r} (expected 'full' or 'gold9').")
    template = (
        PROMPT_SENSE_COUPLING if variant == "v1" else PROMPT_SENSE_COUPLING_V2
    )
    return template.replace("{{sense_menu}}", sense_menu)


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
