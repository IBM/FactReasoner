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

"""Build the LCS example dataset under ``data/lcs/``.

Each example is one response for which we mine inter-atom relations to assess its
Logical Coherence Score. The responses and their atomic-unit decompositions are
transcribed verbatim from the ideation worked examples
(``docs/ideation/example-*-*.tex`` and ``coherence_modeling.tex`` for AeroParts).

For examples whose coherence story hinges on a *variant* (a contradiction-injected
rewrite, or a reordered summary), each variant is emitted as its own file, since
each is a distinct response to be scored.

Output JSON schema (one file per response)::

    {
      "id":          "example-2-biography-contradicted",
      "name":        "Biography (with planted contradictions)",
      "source":      "docs/ideation/example-2-biography.pdf",
      "response":    "<the full response text>",
      "num_atoms":   12,
      "atoms": [ {"id": "a0", "text": "...", "label": "M1"}, ... ],
      "notes": "..."
    }

``atoms[i].id`` is ``a{i}`` (0-based) to match FactReasoner's ``build_atoms`` /
``RelationMiner.mine_from_atoms`` convention; ``label`` preserves the original
doc tag (F1 / M2 / L3 / a1 ...) for traceability to the ``.tex`` sources.

Run::

    python scripts/build_lcs_examples.py            # writes data/lcs/*.json
    python scripts/build_lcs_examples.py --check     # validate only, no write
"""

import argparse
import json
import os
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Example definitions. Each is (metadata, atoms-by-doc-label, gold relations).
# Atoms are listed in source order with their original doc label. Gold relations
# reference the doc labels; they are expanded to a0-based ids at build time.
# ---------------------------------------------------------------------------


def _pre(*pairs):
    """Prerequisite (A -> B, entailment) gold relations from label pairs."""
    return [(a, b, "entailment", "prerequisite") for a, b in pairs]


def _inv(*pairs):
    """Invalidation (A != B, contradiction) gold relations from label pairs."""
    return [(a, b, "contradiction", "invalidation") for a, b in pairs]


EXAMPLES = []


def add_example(id, name, source, response, atoms, gold, notes):
    """Register one example. ``atoms`` is a list of (label, text); ``gold`` is a
    list of (src_label, trg_label, level1_type, relation_class)."""
    EXAMPLES.append(
        dict(
            id=id,
            name=name,
            source=source,
            response=" ".join(response.split()),
            atoms=atoms,
            gold=gold,
            notes=notes,
        )
    )


# --- Example 1: legal damages paragraph ------------------------------------

add_example(
    id="example-1-damages",
    name="Legal damages paragraph",
    source="docs/ideation/example-1-damages.pdf",
    response="""
    The defendant was found liable for breach of contract, which occurred on
    March 15, 2022, after they failed to deliver the specified goods. As a
    result, the court awarded the plaintiff $50,000 in damages for the breach.
    However, the defendant argued that the plaintiff's own actions constituted a
    prior breach, thus excusing their performance. The case hinged on the initial
    agreement signed on January 1, 2022, which the court ultimately declared void
    ab initio (void from the beginning) due to fraudulent misrepresentation by
    the plaintiff. Separately, the court found the defendant liable for unjust
    enrichment and awarded the plaintiff $10,000 on that basis, as the defendant
    had retained a down payment.
    """,
    atoms=[
        ("F1", "The defendant was found liable for breach of contract."),
        ("F2", "The breach of contract occurred on March 15, 2022."),
        ("F3", "The defendant failed to deliver the specified goods."),
        ("F4", "The court awarded the plaintiff $50,000 in damages for the breach."),
        ("F5", "The defendant argued that the plaintiff's own actions constituted a prior breach."),
        ("F6", "The defendant argued that the plaintiff's prior breach excused the defendant's performance."),
        ("F7", "The case hinged on the initial agreement."),
        ("F8", "The initial agreement was signed on January 1, 2022."),
        ("F9", "The court declared the initial agreement void ab initio (void from the beginning)."),
        ("F10", "The agreement was declared void due to fraudulent misrepresentation by the plaintiff."),
        ("F11", "The court found the defendant liable for unjust enrichment."),
        ("F12", "The court awarded the plaintiff $10,000 on the basis of unjust enrichment."),
        ("F13", "The defendant had retained a down payment."),
    ],
    gold=(
        _pre(("F3", "F1"), ("F1", "F2"), ("F1", "F4"), ("F7", "F1"), ("F7", "F8"),
             ("F7", "F9"), ("F8", "F9"), ("F10", "F9"), ("F13", "F11"), ("F11", "F12"))
        + _inv(("F9", "F1"), ("F9", "F2"), ("F9", "F4"), ("F9", "F6"), ("F10", "F5"))
    ),
    notes="Void-ab-initio pivot (F9) retroactively invalidates the breach cluster; "
          "the unjust-enrichment chain (F11-F13) survives.",
)


# --- Example 2a: biography (consistent) ------------------------------------

add_example(
    id="example-2-biography",
    name="Biography (consistent)",
    source="docs/ideation/example-2-biography.pdf",
    response="""
    Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola,
    Florida. He has appeared in numerous films, television shows, and theater
    productions throughout his career, which began in the late 1970s. Some of his
    notable film credits include "King of New York," "The Abyss," "Natural Born
    Killers," "The Game," and "The Straight Story." On television, he has appeared
    in shows such as "Law & Order," "The Sopranos," "Boardwalk Empire," and "The
    Leftovers." Flaherty has also worked extensively in theater, including
    productions at the Public Theater and the New York Shakespeare Festival. He is
    known for his distinctive looks and deep gravelly voice, which have made him a
    memorable character actor in the industry.
    """,
    atoms=[
        ("F1", "Lanny Flaherty is an American actor."),
        ("F2", "Lanny Flaherty was born on December 18, 1949."),
        ("F3", "Lanny Flaherty was born in Pensacola, Florida."),
        ("F4", "Lanny Flaherty has appeared in numerous films, television shows, and theater productions."),
        ("F5", "Lanny Flaherty's career began in the late 1970s."),
        ("F6", "Lanny Flaherty's notable film credits include \"King of New York.\""),
        ("F7", "Lanny Flaherty's notable film credits include \"The Abyss.\""),
        ("F8", "Lanny Flaherty's notable film credits include \"Natural Born Killers.\""),
        ("F9", "Lanny Flaherty's notable film credits include \"The Game.\""),
        ("F10", "Lanny Flaherty's notable film credits include \"The Straight Story.\""),
        ("F11", "Lanny Flaherty has appeared in the television show \"Law & Order.\""),
        ("F12", "Lanny Flaherty has appeared in the television show \"The Sopranos.\""),
        ("F13", "Lanny Flaherty has appeared in the television show \"Boardwalk Empire.\""),
        ("F14", "Lanny Flaherty has appeared in the television show \"The Leftovers.\""),
        ("F15", "Lanny Flaherty has worked extensively in theater."),
        ("F16", "Lanny Flaherty's theater work includes productions at the Public Theater."),
        ("F17", "Lanny Flaherty's theater work includes productions at the New York Shakespeare Festival."),
        ("F18", "Lanny Flaherty is known for his distinctive looks and deep gravelly voice."),
        ("F19", "Lanny Flaherty's distinctive looks and voice have made him a memorable character actor."),
    ],
    gold=_pre(
        ("F1", "F4"), ("F1", "F5"),
        ("F4", "F6"), ("F4", "F7"), ("F4", "F8"), ("F4", "F9"), ("F4", "F10"),
        ("F4", "F11"), ("F4", "F12"), ("F4", "F13"), ("F4", "F14"), ("F4", "F15"),
        ("F15", "F16"), ("F15", "F17"), ("F18", "F19"), ("F1", "F19"),
    ),
    notes="Internally consistent: no invalidation edges (the absence is itself the finding).",
)


# --- Example 2b: biography with planted contradictions ---------------------

add_example(
    id="example-2-biography-contradicted",
    name="Biography (with planted contradictions)",
    source="docs/ideation/example-2-biography.pdf",
    response="""
    Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola,
    Florida. His career began in the late 1970s, and he appeared in numerous
    films, television shows, and theater productions. He is known for his deep
    gravelly voice. Born in 1942, Flaherty was a Mississippi native who first
    appeared on screen only in the 1990s. Critics often described the British
    character actor for his soft, high-pitched voice, which made him instantly
    recognizable.
    """,
    atoms=[
        ("M1", "Lanny Flaherty is an American actor."),
        ("M2", "Lanny Flaherty was born on December 18, 1949."),
        ("M3", "Lanny Flaherty was born in Pensacola, Florida."),
        ("M4", "Lanny Flaherty's career began in the late 1970s."),
        ("M5", "Lanny Flaherty appeared in numerous films, television shows, and theater productions."),
        ("M6", "Lanny Flaherty is known for his deep gravelly voice."),
        ("M7", "Lanny Flaherty was born in 1942."),
        ("M8", "Lanny Flaherty was a Mississippi native."),
        ("M9", "Lanny Flaherty first appeared on screen only in the 1990s."),
        ("M10", "Lanny Flaherty is a British character actor."),
        ("M11", "Lanny Flaherty has a soft, high-pitched voice."),
        ("M12", "Lanny Flaherty's voice made him instantly recognizable."),
    ],
    gold=(
        _pre(("M1", "M5"), ("M1", "M4"), ("M11", "M12"))
        + _inv(("M2", "M7"), ("M3", "M8"), ("M4", "M9"), ("M1", "M10"), ("M6", "M11"))
    ),
    notes="Five planted contradictions: each later atom (M7-M11) contradicts an "
          "earlier one asserted at a different position (order-sensitive invalidation).",
)


# --- Example 3: narrative passage (Elinor) ---------------------------------

add_example(
    id="example-3-narrative",
    name="Narrative passage (Elinor)",
    source="docs/ideation/example-3-narrative.pdf",
    response="""
    Elinor is surprised no letter arrives with details about the marriage, and
    she wonders how Edward felt being close to Barton. Elinor asks her mother to
    write to Colonel Brandon for news, realizing she hoped something would prevent
    Edward's marriage. Elinor feels incredibly hurt by Edward's marriage. Colonel
    Brandon arrives at the house, but it is revealed to be Edward instead. Elinor
    is surprised that Edward and Lucy were married so soon, and she mulls over the
    event of Edward's marriage, supposing that the Ferrars must be settled at
    Delaford, envisioning Edward and Lucy's life there. Edward enters the room
    looking ill with anxiety, and Elinor tells herself to be calm upon Edward's
    arrival. Elinor thinks someone in London might have informed her about Edward
    and Lucy's marriage. Marianne and Margaret retreat, leaving Elinor and
    Mrs. Dashwood with Edward, while Marianne and Mrs. Dashwood react with shock
    and uncertainty about Edward's arrival. Mrs. Dashwood breaks the ice by
    shaking Edward's hand and wishing him happiness. However, no word has arrived
    from Colonel Brandon, and Mrs. Dashwood asks about Mrs. Ferrars' health, to
    which Elinor inquires if Mrs. Ferrars is at Longstaple. Edward is shocked to
    discover his mother is not nearby, and Elinor pointedly asks about Mrs. Edward
    Ferrars. Edward is confused and asks if Elinor means Mrs. Robert Ferrars
    instead, leading to everyone being shocked by the revelation that Robert
    married Lucy Steele. Elinor flees the room, unsure of how to react, and bursts
    into tears of joy. Edward sits in silence, unsure of what to do, then simply
    leaves the room. Everyone is left perplexed by the situation.
    """,
    atoms=[
        ("F1", "Elinor is surprised that no letter arrives with details about the marriage."),
        ("F2", "Elinor wonders how Edward felt being close to Barton."),
        ("F3", "Elinor asks her mother to write to Colonel Brandon for news."),
        ("F4", "Elinor realizes she had hoped something would prevent Edward's marriage."),
        ("F5", "Elinor feels incredibly hurt by Edward's marriage."),
        ("F6", "A visitor arrives at the house and is initially taken to be Colonel Brandon."),
        ("F7", "The visitor is revealed to be Edward instead of Colonel Brandon."),
        ("F8", "Elinor is surprised that Edward and Lucy were married so soon."),
        ("F9", "Elinor mulls over the event of Edward's marriage."),
        ("F10", "Elinor supposes that the Ferrars must be settled at Delaford."),
        ("F11", "Elinor envisions Edward and Lucy's life at Delaford."),
        ("F12", "Edward enters the room looking ill with anxiety."),
        ("F13", "Elinor tells herself to be calm upon Edward's arrival."),
        ("F14", "Elinor thinks someone in London might have informed her about Edward and Lucy's marriage."),
        ("F15", "Marianne and Margaret retreat from the room."),
        ("F16", "Marianne and Margaret leave Elinor and Mrs. Dashwood alone with Edward."),
        ("F17", "Marianne and Mrs. Dashwood react with shock and uncertainty about Edward's arrival."),
        ("F18", "Mrs. Dashwood breaks the ice by shaking Edward's hand."),
        ("F19", "Mrs. Dashwood wishes Edward happiness."),
        ("F20", "No word has arrived from Colonel Brandon."),
        ("F21", "Mrs. Dashwood asks about Mrs. Ferrars' health."),
        ("F22", "Elinor inquires whether Mrs. Ferrars is at Longstaple."),
        ("F23", "Edward is shocked to discover Edward's mother is not nearby."),
        ("F24", "Elinor pointedly asks about Mrs. Edward Ferrars."),
        ("F25", "Edward is confused by Elinor's question about Mrs. Edward Ferrars."),
        ("F26", "Edward asks whether Elinor means Mrs. Robert Ferrars instead."),
        ("F27", "It is revealed that Robert married Lucy Steele."),
        ("F28", "Everyone is shocked by the revelation that Robert married Lucy Steele."),
        ("F29", "Elinor flees the room, unsure of how to react."),
        ("F30", "Elinor bursts into tears of joy."),
        ("F31", "Edward sits in silence, unsure of what to do."),
        ("F32", "Edward then leaves the room."),
        ("F33", "Everyone is left perplexed by the situation."),
    ],
    gold=(
        _pre(("F6", "F7"), ("F7", "F12"), ("F9", "F10"), ("F10", "F11"),
             ("F15", "F16"), ("F22", "F23"), ("F24", "F25"), ("F25", "F26"),
             ("F26", "F27"), ("F27", "F28"), ("F27", "F30"))
        + _inv(("F27", "F1"), ("F27", "F4"), ("F27", "F5"), ("F27", "F8"),
               ("F27", "F9"), ("F27", "F10"), ("F27", "F11"), ("F27", "F14"))
    ),
    notes="Late revelation F27 (Robert, not Edward, married Lucy) retroactively "
          "invalidates the 'Edward married Lucy' premise woven through the opening.",
)


# --- Example 4: synthesized summary S --------------------------------------

add_example(
    id="example-4-summary",
    name="Synthesized summary S (reliable + unreliable sources)",
    source="docs/ideation/example-4-summary.pdf",
    response="""
    Keyara Jones is an American college basketball player from Montgomery,
    Alabama, who plays guard for the University of Alabama. During the 2020-2021
    season she made history as the first woman to join an NCAA Division I men's
    basketball team, averaging 18 points per game and leading Alabama to a Sweet
    16 appearance. Because she competed in the men's league, her achievement was
    hailed as a milestone for gender equality, and NCAA data showing that 12% of
    Division I men's players are women was cited to underscore the trend. In fact,
    the University of Alabama's athletic website lists Jones as a guard from
    Birmingham, Alabama, who was a member of the women's basketball team during
    the 2020-2021 season. The Crimson Tide women's team competes in the
    Southeastern Conference of NCAA Division I women's basketball, which is the
    highest level of women's college basketball in the United States and consists
    of 356 teams.
    """,
    atoms=[
        ("S1", "Keyara Jones is an American college basketball player."),
        ("S2", "Keyara Jones is from Montgomery, Alabama."),
        ("S3", "Keyara Jones plays guard for the University of Alabama."),
        ("S4", "During the 2020-2021 season Keyara Jones made history as the first woman to join an NCAA Division I men's basketball team."),
        ("S5", "Keyara Jones averaged 18 points per game."),
        ("S6", "Keyara Jones led Alabama to a Sweet 16 appearance."),
        ("S7", "Keyara Jones competed in the men's basketball league."),
        ("S8", "Keyara Jones' achievement was hailed as a milestone for gender equality."),
        ("S9", "NCAA data show that 12% of Division I men's basketball players are women."),
        ("S10", "The 12% statistic was cited to underscore the gender-equality trend."),
        ("S11", "The University of Alabama's athletic website lists Keyara Jones as a guard from Birmingham, Alabama."),
        ("S12", "Keyara Jones was a member of the University of Alabama women's basketball team during the 2020-2021 season."),
        ("S13", "The Crimson Tide women's team competes in the Southeastern Conference of NCAA Division I women's basketball."),
        ("S14", "NCAA Division I women's basketball is the highest level of women's college basketball in the United States."),
        ("S15", "NCAA Division I women's basketball consists of 356 teams."),
    ],
    gold=(
        _pre(("S3", "S4"), ("S4", "S7"), ("S7", "S8"), ("S9", "S10"),
             ("S12", "S13"), ("S13", "S14"), ("S14", "S15"), ("S1", "S3"))
        + _inv(("S11", "S2"), ("S12", "S4"), ("S12", "S7"), ("S12", "S8"),
               ("S12", "S10"))
    ),
    notes="Reliable atoms (S11, S12) stated after B's false framing positionally "
          "invalidate the men's-league thread (S4-S10). S9 is also self-inconsistent.",
)


# --- Example 5a: R v Renda, summary S (adversarial ordering) ---------------

add_example(
    id="example-5-renda-S",
    name="R v Renda summary S (self-serving-first ordering)",
    source="docs/ideation/example-5-renda.pdf",
    response="""
    Raymond Renda was convicted of attempted robbery and appealed against that
    conviction. The charge arose when, at about 2 am on 10 November 2003, Renda
    accosted Robert Flint on Mile End Road, followed him home, and seized him by
    the neck demanding money; the trial turned on the relative credibility of
    Flint and Renda. To enhance his credibility Renda testified to his good
    character: he claimed that his serious head injury had been sustained while he
    was serving on duty in the armed forces, and that he was currently in regular
    employment as a security guard. In fact, as he conceded under
    cross-examination, the head injury was sustained on holiday in a car accident,
    and his security work had been only short-term pass-checking so that he was no
    longer employed; he had therefore conveyed a false impression of positive good
    character. The defence argued that this cross-examination concession withdrew
    the false impression under s 105(3), but the court held that a concession
    forced in cross-examination is not a withdrawal. The court also recounted that
    in July 2001 Renda had been found unfit to plead to assault and given an
    absolute discharge, so that he stood not convicted, even though a jury had
    found as a fact that he struck a man from behind with a wooden table leg,
    which the court held to be reprehensible behaviour. Renda's counsel initially
    conceded that this table-leg finding amounted to a conviction, then later
    concluded that it was not a conviction. The court held that the evidence was
    not contaminated, and the appeal was dismissed.
    """,
    atoms=[
        ("L1", "Raymond Renda was convicted of attempted robbery."),
        ("L2", "Raymond Renda appealed against the attempted-robbery conviction."),
        ("L3", "Renda accosted Robert Flint on Mile End Road at about 2 am on 10 November 2003 and seized him by the neck while demanding money."),
        ("L4", "The trial turned on the relative credibility of Robert Flint and Raymond Renda."),
        ("L5", "Renda testified to his good character in order to enhance his credibility."),
        ("L6", "Renda claimed his serious head injury was sustained while he was serving on duty in the armed forces."),
        ("L7", "Renda claimed he was currently in regular employment as a security guard."),
        ("L8", "In fact Renda's serious head injury was sustained on holiday in a car accident."),
        ("L9", "In fact Renda's security work had been only short-term pass-checking and he was no longer employed."),
        ("L10", "Renda had conveyed a false impression of positive good character."),
        ("L11", "The defence argued that Renda's cross-examination concession withdrew the false impression under s 105(3)."),
        ("L12", "The court held that a concession forced in cross-examination is not a withdrawal of a false impression."),
        ("L13", "In July 2001 Renda was found unfit to plead to assault and given an absolute discharge, so he stood not convicted."),
        ("L14", "A jury found as a fact that Renda struck a man from behind with a wooden table leg."),
        ("L15", "The court held that the table-leg incident was reprehensible behaviour."),
        ("L16", "Renda's counsel initially conceded that the table-leg finding amounted to a conviction."),
        ("L17", "Renda's counsel later concluded that the table-leg finding was not a conviction."),
        ("L18", "The court held that the evidence was not contaminated and dismissed the appeal."),
    ],
    gold=(
        _pre(("L1", "L2"), ("L3", "L1"), ("L4", "L5"), ("L5", "L6"), ("L5", "L7"),
             ("L8", "L10"), ("L9", "L10"), ("L10", "L11"), ("L13", "L15"),
             ("L14", "L15"), ("L12", "L18"), ("L15", "L18"))
        + _inv(("L8", "L6"), ("L9", "L7"), ("L12", "L11"), ("L15", "L13"),
               ("L17", "L16"))
    ),
    notes="Self-serving claims stated first, conceded truths/holdings after, so the "
          "later atoms positionally contradict the earlier ones. Contrast with K.",
)


# --- Example 5b: R v Renda, summary K (faithful natural ordering) ----------

add_example(
    id="example-5-renda-K",
    name="R v Renda summary K (faithful natural ordering)",
    source="docs/ideation/example-5-renda.pdf",
    response="""
    Raymond Renda was convicted of attempted robbery before HHJ Van Der Werff and
    a jury, and appealed. The charge arose when, at about 2 am on 10 November
    2003, he accosted Robert Flint on Mile End Road, followed him home, and seized
    him by the neck while demanding money; because the only real question was
    whether any offence had been committed, the trial turned on the relative
    credibility of Flint and Renda. Seeking to present himself as a man of
    positive good character, Renda testified that his serious head injury had been
    sustained on duty in the armed forces and that he was in regular employment as
    a security guard; in truth, however, the injury was sustained on holiday in a
    car accident and the security work had been only short-term, so that this
    evidence conveyed a false impression within s 105(1). When Renda conceded
    these matters under cross-examination, the court held that a concession
    extracted in cross-examination does not amount to a withdrawal of the false
    impression under s 105(3). To help correct that false impression, the Crown
    was permitted to put a July 2001 incident before the jury: Renda had then been
    found unfit to plead to assault and given an absolute discharge, so he was not
    convicted, yet a jury had found as a fact that he struck a man from behind with
    a wooden table leg, and the court held that this act was reprehensible
    behaviour within the bad-character provisions notwithstanding the absence of a
    conviction. A submission that the resulting evidence was contaminated under
    s 107 was rejected, the court holding that it was not contaminated, and the
    appeal was dismissed.
    """,
    atoms=[
        ("K1", "Raymond Renda was convicted of attempted robbery and appealed against the conviction."),
        ("K2", "Renda accosted Robert Flint on Mile End Road at about 2 am on 10 November 2003, followed him home, and seized him by the neck while demanding money."),
        ("K3", "The trial turned on the relative credibility of Robert Flint and Raymond Renda."),
        ("K4", "Renda testified seeking to present himself as a man of positive good character."),
        ("K5", "Renda testified that his head injury was sustained on duty in the armed forces and that he was in regular employment as a security guard."),
        ("K6", "In truth Renda's injury was sustained on holiday in a car accident and his security work had been only short-term, so his evidence conveyed a false impression within s 105(1)."),
        ("K7", "Renda conceded these matters under cross-examination."),
        ("K8", "The court held that a concession extracted in cross-examination does not amount to a withdrawal of the false impression under s 105(3)."),
        ("K9", "To help correct the false impression, the Crown was permitted to put a July 2001 incident before the jury."),
        ("K10", "In the July 2001 incident Renda was found unfit to plead to assault and given an absolute discharge, so he was not convicted."),
        ("K11", "A jury had found as a fact that Renda struck a man from behind with a wooden table leg."),
        ("K12", "The court held that the table-leg act was reprehensible behaviour within the bad-character provisions, notwithstanding the absence of a conviction."),
        ("K13", "A submission was made that the evidence was contaminated under s 107."),
        ("K14", "The court held that the evidence was not contaminated."),
        ("K15", "The appeal was dismissed."),
    ],
    gold=(
        _pre(("K2", "K1"), ("K1", "K3"), ("K3", "K4"), ("K4", "K5"), ("K5", "K6"),
             ("K6", "K7"), ("K7", "K8"), ("K6", "K9"), ("K10", "K12"),
             ("K11", "K12"), ("K9", "K13"), ("K13", "K14"), ("K8", "K15"),
             ("K12", "K15"), ("K14", "K15"))
    ),
    notes="Same facts as S but faithfully ordered: each disputed claim is folded into "
          "its finding (holdings K8/K12 resolve the tensions), so there are NO "
          "positional invalidations. Should score markedly higher than S.",
)


# --- AeroParts recall report (the deep-dive running example) ---------------

add_example(
    id="aeroparts-recall",
    name="AeroParts turbine-blade recall report",
    source="docs/ideation/coherence_mrf_deepdive.pdf",
    response="""
    AeroParts shipped a batch of turbine blades certified to spec, and the blades
    were installed in the carrier's A-320 fleet. The blades then entered regular
    service across the fleet. During service, seventeen blades failed in flight.
    The regulator opened a safety probe, and within a week issued an order
    grounding the fleet; in effect, every affected aircraft was pulled from
    service. Maintenance logs, later disclosed, recorded abnormal vibration on the
    failed units. The airline stated that no passengers or crew were harmed in any
    of the incidents. A separate wire report, however, claimed that three
    passengers died in one of the failures. Early in the inquiry the airline
    argued that the failures were caused by pilot error. The final accident report
    concluded that the root cause was a metallurgical defect in the AeroParts
    blades, and on that basis found the pilots not at fault. Because AeroParts had
    shipped the defective batch, the tribunal named AeroParts liable and ordered
    it to pay damages. The regulator also published a preliminary bulletin.
    """,
    atoms=[
        ("a1", "AeroParts shipped a batch of turbine blades certified to spec."),
        ("a2", "The turbine blades were installed in the carrier's A-320 fleet."),
        ("a3", "The turbine blades entered regular service across the fleet."),
        ("a4", "Seventeen turbine blades failed in flight during service."),
        ("a5", "The regulator opened a safety probe into the blade failures."),
        ("a6", "The regulator issued an order grounding the fleet."),
        ("a7", "The airline stated that no passengers or crew were harmed in the incidents."),
        ("a8", "Every affected aircraft was pulled from service."),
        ("a9", "Maintenance logs recorded abnormal vibration on the failed blades."),
        ("a10", "A wire report claimed that three passengers died in one of the failures."),
        ("a11", "The airline argued that the failures were caused by pilot error."),
        ("a12", "The final accident report concluded the root cause was a metallurgical defect in the AeroParts blades."),
        ("a13", "The accident report found the pilots not at fault."),
        ("a14", "The tribunal named AeroParts liable."),
        ("a15", "The tribunal ordered AeroParts to pay damages."),
        ("a16", "The regulator published a preliminary bulletin."),
    ],
    gold=(
        # Entailment / evidence / restatement spine (deep-dive Section 5 table).
        [("a1", "a2", "entailment", "prerequisite"),
         ("a2", "a3", "entailment", "prerequisite"),
         ("a3", "a4", "entailment", "prerequisite"),
         ("a4", "a5", "entailment", "prerequisite"),
         ("a5", "a6", "entailment", "prerequisite"),
         ("a4", "a7", "entailment", "prerequisite"),
         ("a9", "a4", "entailment", "prerequisite"),
         ("a6", "a8", "equivalence", "restatement"),
         ("a1", "a14", "entailment", "prerequisite"),
         ("a14", "a15", "entailment", "prerequisite"),
         ("a5", "a16", "entailment", "prerequisite"),
         ("a13", "a12", "entailment", "prerequisite")]
        # Contradictions: a7!=a10 unresolved; a11!=a12 concession resolved by a13.
        + _inv(("a7", "a10"), ("a11", "a12"))
    ),
    notes="Deep-dive running example. Unresolved casualty conflict a7!=a10 ('no one "
          "harmed' vs 'three died'); blame tension a11!=a12 is a concession resolved "
          "by holding a13. Exact LCS: 0.587 (base) -> 0.620 (contradictions removed).",
)


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------


def _build_record(example: Dict) -> Dict:
    """Expand one registered example into the output JSON record."""
    atoms = [
        {"id": f"a{i}", "text": text, "label": label}
        for i, (label, text) in enumerate(example["atoms"])
    ]

    return {
        "id": example["id"],
        "name": example["name"],
        "source": example["source"],
        "response": example["response"],
        "num_atoms": len(atoms),
        "atoms": atoms,
        "notes": example["notes"],
    }


def _validate(record: Dict) -> None:
    """Sanity-check a built record (ids contiguous a0.., non-empty response)."""
    ids = [a["id"] for a in record["atoms"]]
    expected = [f"a{i}" for i in range(len(ids))]
    assert ids == expected, f"{record['id']}: atom ids not contiguous a0..: {ids}"
    assert record["response"], f"{record['id']}: empty response"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <repo>/data/lcs).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the built records without writing files.",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.out_dir or os.path.join(repo_root, "data", "lcs")

    records = [_build_record(ex) for ex in EXAMPLES]
    for rec in records:
        _validate(rec)

    if args.check:
        for rec in records:
            print(f"OK  {rec['id']:34s} atoms={rec['num_atoms']:2d}")
        print(f"\n{len(records)} examples validated (no files written).")
        return

    os.makedirs(out_dir, exist_ok=True)
    for rec in records:
        path = os.path.join(out_dir, f"{rec['id']}.json")
        with open(path, "w") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"wrote {path}  (atoms={rec['num_atoms']})")

    _write_readme(out_dir, records)


def _write_readme(out_dir: str, records: List[Dict]) -> None:
    """Write a short README describing the dataset schema and contents."""
    lines = [
        "# LCS example dataset (`data/lcs/`)",
        "",
        "Example responses for mining inter-atom relations and assessing the",
        "Logical Coherence Score (LCS). Each JSON holds one response and its",
        "atomic-unit decomposition, transcribed from the ideation worked examples",
        "(`docs/ideation/example-*-*.pdf`; AeroParts from `coherence_mrf_deepdive.pdf`).",
        "",
        "## Schema",
        "",
        "```json",
        "{",
        '  "id": "example-1-damages",',
        '  "name": "...", "source": "docs/ideation/....pdf",',
        '  "response": "<full response text>",',
        '  "num_atoms": 13,',
        '  "atoms": [{"id": "a0", "text": "...", "label": "F1"}, ...],',
        '  "notes": "..."',
        "}",
        "```",
        "",
        "- `atoms[i].id` is `a{i}` (0-based), matching `RelationMiner.mine_from_atoms`",
        "  and `build_atoms`. `label` is the original doc tag (F/M/L/S/K/a).",
        "",
        "## Usage",
        "",
        "```python",
        "import json",
        "from fact_reasoner import build_backend, RelationMiner, LCSScorer",
        "",
        'ex = json.load(open("data/lcs/aeroparts-recall.json"))',
        'atoms = [a["text"] for a in ex["atoms"]]',
        "",
        'backend = build_backend("rits", model_id="llama-3-3-70b-instruct")',
        "miner = RelationMiner(backend, pair_policy=\"all_pairs\")",
        "result = miner.mine_from_atoms(atoms)",
        "scores = LCSScorer(merlin_path).score(result)",
        "```",
        "",
        "## Files",
        "",
    ]
    for rec in sorted(records, key=lambda r: r["id"]):
        lines.append(
            f"- `{rec['id']}.json` — {rec['name']} ({rec['num_atoms']} atoms)"
        )
    lines.append("")
    path = os.path.join(out_dir, "README.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
