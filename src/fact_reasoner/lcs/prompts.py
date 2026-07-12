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

# LLM prompts for atom-atom relation mining (deep-dive Sections 4.2-4.3).
#
# Two prompts implement the type-posterior x conditional-strength decomposition
# p = P(tau | a_i, a_j) x P(a_j | a_i, tau):
#
#   * Prompt A (PROMPT_SENSE_COUPLING) -- a chain-of-thought call over an ordered
#     atom pair (A, B) that names the Level-2 discourse SENSE and maps it to a
#     Level-1 COUPLING. The final answer is a bracketed [coupling=...] tag whose
#     token logprobs give the type confidence P(tau | a_i, a_j). The chain of
#     thought comes first so the model commits to the sense only after reasoning.
#
#   * Prompt B -- given the coupling from Prompt A, elicits the conditional
#     strength P(a_j | a_i, tau). TWO forms are provided:
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
    "Instantiation, Contrast, Concession, Precedence, Succession, None"
)


# ----------------------------------------------------------------------------
# Prompt A -- joint discourse sense + Level-1 coupling classification.
# ----------------------------------------------------------------------------

PROMPT_SENSE_COUPLING = """

Instructions:
You are given two atomic claims, A and B, taken from the SAME response, in \
their order of appearance (A comes before B). Your task is to decide the \
discourse/logical relation FROM A TO B, following the steps below.

1. Reason step by step: does A cause, enable, provide evidence for, restate, \
elaborate, temporally precede, contrast with, or contradict B? Is B a claim the \
text later withdraws or that a holding resolves? Consider the direction (A to B).

2. Name the DISCOURSE SENSE, one of: {{sense_menu}}.
   - Cause-Effect: A causes/leads to B. Effect-Cause: A is the effect, B its cause.
   - Evidence: A provides evidence for B. Condition: A is a condition for B.
   - Restatement: A and B assert the same thing. Instantiation: A is a general \
claim, B a specific instance (or vice versa).
   - Contrast: A and B are in opposition. Concession: A and B are in tension but \
the text concedes/resolves it ("although A, still B", or a holding settles it).
   - Precedence/Succession: A and B are ordered in time with no truth dependence.
   - None: no logical or discourse dependence between A and B.

3. Map the sense to a COUPLING, one of: entailment, contradiction, \
equivalence, none.
   - Cause-Effect, Effect-Cause, Evidence, Condition, Instantiation -> entailment
   - Restatement -> equivalence
   - Contrast, Concession -> contradiction
   - Precedence, Succession, None -> none

4. Give your final answer as two bracketed tags on ONE line, sense first:
[sense=Cause-Effect] [coupling=entailment]
A JSON object {"sense":"Cause-Effect","coupling":"entailment"} is also acceptable.

Use the following examples to better understand your task.

Example 1:
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
1. Reasoning: A describes a flawed product launch; B reports a stock decline in \
the same quarter. A launching a flawed product plausibly causes a stock decline, \
so the relation from A to B is causal (though not certain).
2. Discourse sense: Cause-Effect.
3. Coupling: A causing B is a positive inferential link, i.e. entailment.
4. Final answer:
[sense=Cause-Effect] [coupling=entailment]

Example 2:
A: The new alloy passed every stress test in the recall report.
B: The new alloy is essentially the same material as the previous alloy.
1. Reasoning: A and B are about the same alloy but make different claims; B \
restates neither the pass nor adds a cause. They largely say the same thing about \
the alloy's identity/quality, so this is a restatement.
2. Discourse sense: Restatement.
3. Coupling: A and B assert the same thing, i.e. equivalence.
4. Final answer:
[sense=Restatement] [coupling=equivalence]

Example 3:
A: No one was harmed in the incident.
B: Three people died in the incident.
1. Reasoning: A says no one was harmed; B says three people died in the same \
incident. These cannot both be true; A being true makes B false. There is no \
holding or concession that resolves the tension.
2. Discourse sense: Contrast.
3. Coupling: A makes B false, i.e. contradiction.
4. Final answer:
[sense=Contrast] [coupling=contradiction]

Example 4 (Concession -- a contradiction the text itself resolves):
A: The supplier initially denied any responsibility for the defect.
B: The tribunal ultimately held the supplier liable for the defect.
1. Reasoning: A (the supplier's denial) is in tension with B (the holding of \
liability), but B is an adjudicating holding that resolves the tension rather \
than a raw contradiction. This is a conceded/resolved tension, not an unresolved \
conflict.
2. Discourse sense: Concession.
3. Coupling: the tension maps to contradiction, but note it is resolved by the \
holding in B.
4. Final answer:
[sense=Concession] [coupling=contradiction]

Example 5:
A: The quarterly report was published in April.
B: The annual audit was scheduled for December.
1. Reasoning: A and B describe two separate events with no causal, evidential, \
or contradictory dependence; they merely occur at different times.
2. Discourse sense: None.
3. Coupling: no dependence, i.e. none.
4. Final answer:
[sense=None] [coupling=none]

Your task:
A: {{atom_a}}
B: {{atom_b}}
"""


# ----------------------------------------------------------------------------
# Prompt B (default) -- conditional strength via a Yes/No surrogate token.
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
# still means the contradiction is credible.

PROMPT_STRENGTH_SURROGATE = """

Instructions:
Assume claim A is TRUE, and that a {{coupling}} relation holds from A to B. Judge \
whether the relation's implication about B is credible -- i.e. at least plausible / \
more likely than not, NOT whether it is certain.

- entailment or equivalence: given A, is B at least plausibly TRUE (more likely than \
not)?
- contradiction: given A, is B at least plausibly FALSE (more likely than not)?

Answer with a SINGLE WORD, the very first word of your reply: Yes or No.
- Answer "Yes" if the implication is credible/plausible (even if not certain).
- Answer "No" only if the implication is implausible or unsupported.
Do not output anything before the word Yes or No.

Use the following examples to better understand your task.

Example 1 (near-certain entailment):
A: The new alloy is chemically identical to the certified reference alloy.
B: The new alloy meets the certified reference specification.
coupling: entailment
Answer: Yes

Example 2 (weak but plausible entailment -- still Yes):
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
coupling: entailment
Answer: Yes

Example 3 (clear contradiction):
A: No one was harmed in the incident.
B: Three people died in the incident.
coupling: contradiction
Answer: Yes

Example 4 (implausible / unsupported link):
A: The regulator published a preliminary bulletin.
B: The airline redesigned its loyalty program.
coupling: entailment
Answer: No

Your task:
A: {{atom_a}}
B: {{atom_b}}
coupling: {{coupling}}
Answer: """


# ----------------------------------------------------------------------------
# Prompt B (baseline) -- verbalized conditional strength P(a_j | a_i, tau).
# ----------------------------------------------------------------------------

PROMPT_STRENGTH = """

Instructions:
Assume claim A is TRUE. Under a {{coupling}} relation from A to B, estimate how \
strongly A determines B.

- For an entailment or equivalence coupling: how likely is B to be TRUE given A?
- For a contradiction coupling: how likely is B to be FALSE given A?

Answer with a single probability in [0, 1] to two decimals, after one short \
sentence of justification. A near-certain link is close to 1.00; a merely \
plausible link is around 0.60-0.70. End your answer with the probability wrapped \
in brackets on its own, exactly in the form: [p=0.NN]

Example 1:
A: The new alloy is chemically identical to the certified reference alloy.
B: The new alloy meets the certified reference specification.
coupling: entailment
Justification: chemical identity to a certified reference almost guarantees the \
specification is met, so B follows very strongly from A.
[p=0.95]

Example 2:
A: The company launched a flawed product last quarter.
B: The company's stock price fell 15 percent last quarter.
coupling: entailment
Justification: a flawed product can plausibly drive a stock decline, but many \
other factors affect price, so the link is only moderate.
[p=0.65]

Example 3:
A: No one was harmed in the incident.
B: Three people died in the incident.
coupling: contradiction
Justification: if no one was harmed, it is almost certain that B (three deaths) \
is false.
[p=0.93]

Your task:
A: {{atom_a}}
B: {{atom_b}}
coupling: {{coupling}}
"""


def build_sense_coupling_prompt() -> str:
    """Return Prompt A with the sense menu interpolated.

    The atom placeholders ``{{atom_a}}`` / ``{{atom_b}}`` remain for Mellea's
    ``user_variables`` substitution at call time.

    Returns:
        The Prompt A template string.
    """
    return PROMPT_SENSE_COUPLING.replace("{{sense_menu}}", _SENSE_MENU)


def build_surrogate_strength_prompt() -> str:
    """Return the default (surrogate Yes/No) conditional-strength prompt.

    The ``{{atom_a}}`` / ``{{atom_b}}`` / ``{{coupling}}`` placeholders remain for
    Mellea's ``user_variables`` substitution at call time. The answer's first word
    is the surrogate token whose logprobs give the renormalized strength.

    Returns:
        The surrogate-token strength prompt template string.
    """
    return PROMPT_STRENGTH_SURROGATE


def build_strength_prompt() -> str:
    """Return the verbalized (baseline) conditional-strength prompt.

    The ``{{atom_a}}`` / ``{{atom_b}}`` / ``{{coupling}}`` placeholders remain for
    Mellea's ``user_variables`` substitution at call time.

    Returns:
        The verbalized Prompt B template string.
    """
    return PROMPT_STRENGTH
