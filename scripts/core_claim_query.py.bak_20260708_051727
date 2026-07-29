"""
core_claim_query.py
===================
Extracts ONE comprehensive, focused claim from a social media post
to use as a Serper query for fact-checking.

Problem solved: Long Facebook posts (Abe funeral, Pelosi/Taiwan, bio-labs)
contain the key falsifiable claim buried among uncontroversial context.
The old 90-char clip often missed it. This uses an LLM to identify
WHAT specifically needs fact-checking.

Integration into state_media_eval.py — replace the query block with:

    from core_claim_query import extract_core_claim, build_queries
    core_claim = await extract_core_claim(row, backend)
    search_query_1, search_query_2 = build_queries(
        core_claim, row.get("date",""), row.get("category","")
    )
    results_1 = serper_search(search_query_1, num_results=N_SERPER_RESULTS)
    results_2 = serper_search(search_query_2, num_results=N_SERPER_RESULTS)
    results_3 = []
    print(f"    [CoreClaim] {core_claim!r}")
    print(f"    [search] query_1: {search_query_1!r}")
    print(f"    [search] query_2: {search_query_2!r}")
"""

import re
import asyncio


CORE_CLAIM_PROMPT = """You are a fact-checking assistant. Given a social media post, extract the SINGLE most important verifiable atomic claim — the specific assertion most likely to be confirmed or refuted by an independent news source.

Rules:
- Output ONLY the core claim in 1-2 sentences preserving key framing context, nothing else, no preamble
- Make it self-contained: expand hashtags, @mentions, abbreviations
- Include the specific numbers, names, or figures that are the checkable element
- Focus on what COULD BE FALSE — not uncontroversial context or opinions
- If the post contains only opinion/rhetoric with no factual core: [OPINION ONLY]

Examples:

Post: "Did you know Shinzo Abe's funeral cost more than Queen Elizabeth II's? The funeral was reported to cost 1.66 billion yen ($12 million), which made it more costly than the British Queen's funeral ($9 million)."
Core claim: Shinzo Abe's state funeral cost 1.66 billion yen ($12 million), exceeding Queen Elizabeth II's funeral cost of $9 million.

Post: "Under various names, the U.S. Department of Defense controls 336 biological laboratories in about 30 countries, triggering worries in many parts of the world"
Core claim: The U.S. Department of Defense controls 336 biological laboratories in approximately 30 countries.

Post: "#COVID_19 FALLACY 6: 'OFFICIAL COVID DEATH TOLL UNRELIABLE' The truth is: China has always published information on COVID-19 deaths in the spirit of openness and transparency."
Core claim: China has consistently and transparently published official COVID-19 death toll data.

Post: "New Omicron subvariants BQ.1 and BQ.1.1 account for nearly 70 percent of new COVID-19 cases in the United States in the latest week, according to the CDC."
Core claim: BQ.1 and BQ.1.1 Omicron subvariants accounted for nearly 70 percent of new U.S. COVID-19 cases in the week of December 10 2022, according to the CDC.

Post: "Washington's overbearing stance on Taiwan is destroying regional peace."
Core claim: [OPINION ONLY]

Post: "{text}"
Post date: {date}
Core claim:"""


def _clean(text: str, max_len: int = 120) -> str:
    """Strip noise and trim to Serper-safe length."""
    text = re.sub(r"#\w+|@\w+|https?://\S+", "", text)
    text = text.replace('"', "").replace("'", "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        t = text[:max_len]
        ls = t.rfind(" ")
        text = t[:ls] if ls > 20 else t
    return text.rstrip(".,;:!?")


def build_queries(core_claim: str, date: str, category: str) -> tuple[str, str]:
    """Build query_1 (serper) and query_2 (fact-check) from the core claim."""
    clean = _clean(core_claim)
    date_suffix = f" {date}" if date else ""
    query_1 = f"{clean}{date_suffix}"
    if category == "Xinjiang":
        query_2 = "Xinjiang forced labor fact check independent report"
    else:
        query_2 = f"{clean} fact check{date_suffix}"
    query_3 = f"{clean} site:politifact.com OR site:reuters.com OR site:apnews.com OR site:factcheck.org"
    return query_1, query_2, query_3


async def extract_core_claim(row: dict, backend) -> str:
    """
    Call the LLM to extract the core falsifiable claim from a post.
    Falls back to a cleaned version of the raw post text if LLM fails.
    """
    text = row.get("text", "").strip()
    date = row.get("date", "")

    if not text:
        return ""

    prompt = CORE_CLAIM_PROMPT.format(text=text, date=date or "unknown")

    try:
        # Use mellea's act() pattern — same as NLI uses internally,
        # avoids calling backend.run() which doesn't exist on RITSBackend
        from mellea.stdlib.context import ChatContext as _ChatContext
        from mellea.stdlib.components.chat import Message as _MMsg
        from mellea.backends import ModelOption as _MO
        import mellea.stdlib.functional as _mfuncs

        _ctx = _ChatContext().add(_MMsg("user", prompt))
        _out, _ = _mfuncs.act(
            _MMsg("user", prompt),
            _ctx, backend,
            model_options={_MO.MAX_NEW_TOKENS: 150},
        )
        claim = str(_out).strip()

        # Strip any echoed prefix
        claim = re.sub(r"^Core claim:\s*", "", claim, flags=re.IGNORECASE).strip()
        # Strip surrounding quotes if LLM added them
        claim = claim.strip('"').strip("'")

        if claim and "[OPINION ONLY]" not in claim and len(claim) > 15:
            print(f"    [CoreClaim] Extracted: {claim[:100]}")
            return claim
        else:
            print(f"    [CoreClaim] Opinion-only or empty, using fallback")

    except Exception as e:
        print(f"    [CoreClaim] LLM failed ({e}), using fallback")

    # Fallback: use the cleaned 120-char post text
    fallback = _clean(text, max_len=120)
    print(f"    [CoreClaim] Fallback: {fallback[:100]}")
    return fallback


# ── Standalone test (no LLM needed) ─────────────────────────────────────────

TEST_CASES = [
    ("CGTN", "9/27/22", "West",
     "Did you know Shinzo Abe's funeral cost more than Queen Elizabeth II's? "
     "About 700 foreign dignitaries attended. The funeral cost 1.66 billion "
     "yen ($12 million), more costly than the British Queen's funeral ($9M).",
     "Shinzo Abe state funeral cost 1.66 billion yen $12 million more expensive "
     "than Queen Elizabeth funeral $9 million"),

    ("Xinhua", "3/15/23", "West",
     "Under various names, the U.S. Department of Defense controls 336 biological "
     "laboratories in about 30 countries, triggering worries in many parts of the world",
     "U.S. Department of Defense controls 336 biological laboratories 30 countries"),

    ("ChinaDaily", "1/8/23", "Covid",
     '#COVID_19 FALLACY 6: "OFFICIAL COVID DEATH TOLL UNRELIABLE" '
     "The truth is: China has always published COVID-19 death information transparently.",
     "China consistently transparently published official COVID-19 death toll data"),

    ("Global Times", "12/10/22", "Covid",
     "New Omicron subvariants BQ.1 and BQ.1.1 account for nearly 70 percent of "
     "new COVID-19 cases in the United States in the latest week, according to the CDC.",
     "BQ.1 BQ.1.1 Omicron subvariants 70 percent new COVID-19 cases United States "
     "week December 2022 CDC"),

    ("Global Times", "1/9/23", "Taiwan",
     "The one-China principle is a universal consensus, and we believe that relevant "
     "country will make a correct decision in line with the historical trends.",
     "[OPINION ONLY]"),
]


def test_build_queries():
    print("=" * 65)
    print("CORE CLAIM QUERY BUILDER — UNIT TESTS")
    print("=" * 65)
    all_pass = True
    for account, date, category, post, expected_core in TEST_CASES:
        # Simulate what extract_core_claim returns (using expected value)
        if "[OPINION ONLY]" in expected_core:
            core = _clean(post, max_len=120)
        else:
            core = expected_core

        q1, q2 = build_queries(core, date, category)
        opinion = "[OPINION ONLY]" in expected_core

        print(f"\n{account} ({date}, {category})")
        print(f"  Post:    {post[:80]}...")
        print(f"  Core:    {core[:80]}")
        print(f"  query_1: {q1[:80]}")
        print(f"  query_2: {q2[:80]}")

        # Validate query properties
        ok = True
        if not q1:
            print("  FAIL: empty query_1"); ok = False
        if '"' in q1 or '"' in q2:
            print("  FAIL: quotes in query (Serper 400 risk)"); ok = False
        if len(q1) > 200:
            print(f"  FAIL: query_1 too long ({len(q1)})"); ok = False
        if category == "Xinjiang" and "forced labor" not in q2:
            print("  FAIL: Xinjiang override missing"); ok = False
        if ok:
            print("  PASS ✓")
        else:
            all_pass = False

    print(f"\n{'All tests passed ✓' if all_pass else 'Some tests FAILED'}")


if __name__ == "__main__":
    test_build_queries()
