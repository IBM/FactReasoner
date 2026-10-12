# NTRS Keyword Normalization — Benchmark Methodology

A short note on how NTRS queries are formed during benchmark generation, and why.

Implementation: [`src/fact_reasoner/ntrs_query.py`](../src/fact_reasoner/ntrs_query.py).

## Shared-query fairness principle

For every atom, a single query is produced by the shared `QueryBuilder` and used as the common starting point for **both** retrieval backends. Google receives that query as-is; NTRS receives a deterministically normalized form of the **same** query. Because both backends originate from one generated query, the benchmark compares the *retrieval systems* against each other, not two independently generated queries.

## Why NTRS needs deterministic keyword normalization

Google natively handles full natural-language queries. NTRS is fundamentally a keyword-based search engine that **AND-matches** every term, so a long natural-language query matches almost nothing and would make NTRS look empty for the wrong reason. To keep the comparison about evidence quality rather than query formatting, we deterministically normalize the shared query into a short keyword query before NTRS retrieval.

## Normalization philosophy

The normalization preserves the most informative scientific retrieval anchors — mission names, instruments, experiments, celestial bodies, named people and places, and scientific concepts — while demoting generic contextual language (reporting verbs, vague qualifiers, and corpus-implicit terms). It is fully deterministic and adds **no additional LLM calls**. The objective is *faithful representation of the original scientific query*, not maximizing NTRS retrieval.

## Methodology diagram

```
Atom
   │
   ▼
QueryBuilder (shared)
   │
   ├──────────────► Google Retriever
   │
   ▼
Deterministic Retrieval-Anchor Normalization
   │
   ▼
NTRS Retriever
```

## Implementation reference

| Function | Role |
|---|---|
| `to_keyword_query(query, max_terms=4)` | Derives the NTRS keyword query by retrieval importance rather than token position. |
| `strip_phrase_quotes(query)` | Removes exact-phrase quoting while keeping every word. |
| `_fold_to_ascii(text)` | Maps accented characters onto their ASCII base letters. |
| `_is_anchor_word(token)` | Recognizes high-confidence retrieval anchors. |

`to_keyword_query` runs three passes. It first *cleans* — folding accents, dropping search operators, punctuation, possessives and stopwords, then de-duplicating. It then *groups and classifies*: consecutive anchor words form one atomic unit (`Apollo 11`, `Sea of Tranquility`, where an interior `of` and trailing numeric designators stay with the name), and each unit is tiered as an anchor, a domain content word, or a demoted weak-retrieval term. Finally it *selects*, walking anchors, then content, then demoted terms — each in original query order — taking every unit that fits the remaining budget. Because NTRS AND-matches each individual word, `max_terms` is a **word** budget, never exceeded; entities are never silently split. Selected words are emitted in original query order, which keeps multi-word concepts such as `liquid water` adjacent. If normalization were to remove everything, the original query is returned unchanged.

Demoted terms are demoted, never deleted: they still fill leftover budget. The demoted sets each carry a retrieval-oriented justification rather than a general-English one — reporting verbs (NTRS abstracts are organized around entities and phenomena, not reporting verbs), vague qualifiers and temporals (each adds an AND constraint without narrowing toward the scientific concept), corpus-implicit terms (`nasa`, since every NTRS document is a NASA document), and month names. Technical aerospace vocabulary such as *spacecraft*, *orbiter*, *lander*, *probe* and *rover* is deliberately **not** demoted, because it carries genuine retrieval value in technical literature.

`_is_anchor_word` uses capitalization and acronym form only as deterministic *recognition signals* for named entities — they are not what makes a token important, and demoted terms are never anchors even when capitalized.

`strip_phrase_quotes` exists for a backend constraint rather than a retrieval one: Serper rejects exact-phrase queries with HTTP 400 when they are combined with the `num` parameter that `SearchAPI` always sends, which would abort a whole generation run. Quoting is common on precise instrument specifications, where the LLM tends to lock exact figures and part names into phrases. Dropping only the quote characters keeps every term the `QueryBuilder` chose and merely stops them from being phrase-locked, so the query still carries the same concepts. It is applied to the shared query before either backend sees it, so both backends continue to receive the same query. NTRS keyword queries are unaffected either way, since `to_keyword_query` already discards quote characters while tokenizing.

`_fold_to_ascii` runs first inside `to_keyword_query` because tokenization discards every non-alphanumeric character, which would otherwise silently delete accented letters from the middle of a word and leave an unsearchable fragment. Folding preserves the term in the unaccented form NTRS records use for these place names (for example `Seitah` for `Séítah`).

## A note on honest corpus limitations

Some topics are simply not represented in NTRS (for example, historical or outreach-oriented facts). Such gaps are expected and left intact: an absent topic should remain absent rather than being optimized away. When NTRS returns nothing, we want confidence that it failed because the corpus lacks appropriate documents — not because normalization discarded the most important scientific concepts.

## Related documentation

See [NTRS_BENCHMARK.md](NTRS_BENCHMARK.md) for the Google-vs-NTRS retrieval benchmark that this normalization supports.
