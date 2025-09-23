import ast
import regex
from typing import Literal, TypedDict
import traceback

from tenacity import Retrying, RetryError, stop_after_attempt, wait_random_exponential

from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS

COVERAGE_EVALUATOR_BASE_PROMPT = """{prompt_start}You are an AI assistant specialized in assessing the comprehensiveness of question answers with respect to a set of background texts. Your task is to determine which relevant, atomic pieces of information from the background texts are covered in the evaluated answer, and which are not.

Detailed instructions:
1. Carefully read the provided question, background texts, and the evaluated answer to the question.
2. Identify atomic pieces of information from the background texts that are explicitly covered in the evaluated answer. You should only include information that is directly relevant to answering the original question, ignoring any unrelated content. Think step-by-step as you do this, providing brief reasoning under the Reasoning: header.
3. Identify atomic pieces of information from the background texts that are missing from the evaluated answer. Again, only include information that is relevant to answering the original question. Think step-by-step in the same way and briefly explain your reasoning under the shared Reasoning: header.
4. Once you have completed your analysis, output two separate lists of covered and uncovered statements from the background texts. Each atomic statement should be listed as a separate bullet point under '[Covered statements]' and '[Uncovered statements]' headers as appropriate. For each statement, include the list of background text IDs where it appears in the format [1, 5, 14].

Additional guidance:
- An atomic statement is a minimal, self-contained fact that contributes directly to answering the question.
- If the background texts contain conflicting information, treat each distinct fact as a separate atomic statement.

Use the format outlined below to structure your final output:

Final output:

[Covered statements]
- Covered statement #1. [background text IDs]
- Covered statement #2. [background text IDs]
...

[Uncovered statements]
- Uncovered statement #1. [background text IDs]
- Uncovered statement #2. [background text IDs]
...

Original question:
{query}

{background_texts}

Evaluated answer:
{answer}

Reasoning:{prompt_end}"""

COVERAGE_EVALUATOR_PROMPT = """{prompt_start}You are an AI assistant specialized in assessing the comprehensiveness of question answers with respect to a set of background texts. Your task is to determine which relevant, atomic pieces of information from the background texts are covered in the evaluated answer, and which are not.

Detailed instructions:
1. Carefully read the provided question, background texts, and the evaluated answer to the question.
2. Identify atomic pieces of information from the background texts that are explicitly covered in the evaluated answer. You should only include information that is directly relevant to answering the original question, ignoring any unrelated content. Think step-by-step as you do this, providing brief reasoning under the Reasoning: header.
3. Identify atomic pieces of information from the background texts that are missing from the evaluated answer. Again, only include information that is relevant to answering the original question. Think step-by-step in the same way and briefly explain your reasoning under the shared Reasoning: header.
4. Once you have completed your analysis, output two separate lists of covered and uncovered statements from the background texts. Each atomic statement should be listed as a separate bullet point under '[Covered statements]' and '[Uncovered statements]' headers as appropriate. For each statement, include the list of background text IDs where it appears in the format [1, 5, 14].

Additional guidance:
- An atomic statement is a minimal, self-contained fact that contributes directly to answering the question.
- If the background texts contain conflicting information, treat each distinct fact as a separate atomic statement.

Refer to the following examples to understand the task and the output format.

Example 1:

Original question:
What is the maximum range of Airbus A380?

Background text #1:
The wide-body programme led to the introduction of the four-engine A340 in 1991 and the twinjet A330 in 1992. Production of the A340 ended in 2011, while the A330 would be re-engineered as the A330neo (new engine option) in 2018.

The world's largest passenger airliner was introduced by Airbus in 2005; the A380 is a four-engine aircraft with two full-length passenger seating decks. Intended to challenge the dominance of the Boeing 747 in the long-haul market, the A380 was ultimately a money-losing venture for Airbus due to large development costs and limited sales arising from high operating costs, and production ended in December 2021.

Background text #2:
The Airbus A380 is a very large wide-body airliner, developed and produced by Airbus until 2021. It is the world's largest passenger airliner and the only full-length double-deck jet airliner. The full-length double-deck aircraft has a typical seating for 525 passengers, with a maximum certified capacity for 853 passengers. The quadjet is powered by Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans providing a range of 6,000 nmi (11,100 km; 6,900 mi). As of December 2021, the global A380 fleet had completed more than 800,000 flights with no fatalities and no hull losses.

Background text #3:
Airbus started the work on Airbus A380 development in the late 1990s, with the intention of producing an unprecedentedly large passenger airliner. The Airbus A380 made its maiden flight on April 27, 2005 and was introduced into regular service in 2007. The aircraft provides a massive range of approximately 8,000 nautical miles (14,800 km). While it has become an icon of the skies, it was a commercial failure due to large development costs and limited sales.

Evaluated answer:
Airbus A380 has defined the ultra-long-range travel. Its versatility and adaptability to various conditions has contributed to its impressive range of 11100 km.

Reasoning:
(Brief, step-by-step reasoning would be provided here, but is omitted from the example.)

Final output:

[Covered statements]
- The Airbus A380 has a range of 11100 km (6000 nmi; 6900 mi). [2]

[Uncovered statements]
- The Airbus A380 has a range of approximately 8,000 nautical miles (14,800 km). [3]

Example 2:

Original question:
Tell me about Glenn Danzig.

Background text #1:
Beginning in the mid-1970s, Danzig's musical career has encompassed a number of genres through the years, including punk rock and heavy metal, and incorporating influences from industrial, blues and classical music. He has also written songs for other musicians, most notably Johnny Cash and Roy Orbison.

As a singer, Danzig is noted for his baritone voice and tenor vocal range; his style has been compared to those of Elvis Presley, Jim Morrison, and Howlin' Wolf. Danzig has also cited Bill Medley as a vocal influence. In 2023, Rolling Stone ranked Glenn Danzig at number 199 on its list of the 200 Greatest Singers of All Time.

Background text #2:
Glenn Danzig was born on 06/23/1955 in Lodi, New Jersey. In the mid-1970s, Danzig started the Misfits band, but disbanded it in October 1983 due to personal and professional differences. Before disbanding of the Misfits, Danzig had begun working on a new band project, Samhain.

Background text #3:
Glenn Danzig (born in 1955 in Lodi, New Jersey) is an American singer, songwriter, musician, and record producer. He is the founder of the rock band Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.

Evaluated answer:
Glenn Danzig is an American musician, singer, and songwriter best known as the founder of the punk band Misfits, the gothic rock group Samhain, and the heavy metal band Danzig. Born in 1955 in New Jersey, he developed a distinctive baritone voice and dark lyrical style influenced by horror literature and classic rock icons like Elvis Presley and Jim Morrison. Beyond music, Danzig has written songs for artists like Johnny Cash, created the adult-themed comic book company Verotik, and directed horror films. His work has had a lasting impact on punk, metal, and alternative culture.

Reasoning:
(Brief, step-by-step reasoning would be provided here, but is omitted from the example.)

Final output:

[Covered statements]
- Glenn Danzig is an American. [3]
- Glenn Danzig is a singer. [2, 3]
- Glenn Danzig is a songwriter. [1, 3]
- Glenn Danzig is associated with the punk music genre. [1]
- Glenn Danzig is the founder of the band Misfits. [2]
- Glenn Danzig is the founder of the band Samhain. [2]
- Glenn Danzig is associated with the heavy metal music genre. [1]
- Glenn Danzig is the founder of the band Danzig. [3]
- Glenn Danzig was born in 1955. [2, 3]
- Glenn Danzig was born in New Jersey. [2, 3]
- Glenn Danzig has a distinct baritone voice. [1]
- Glenn Danzig's vocal style has been linked to Elvis Presley. [1]
- Glenn Danzig's vocal style has been linked to Jim Morrison. [1]
- Glenn Danzig has written a song for Jonny Cash. [1]
- Gelnn Danzig is associated with the adult-themed comic book company Verotik. [3]

[Uncovered statements]
- Glenn Danzig's music incorporates influences from industrial music. [1]
- Glenn Danzig's music incorporates influences from blues. [1]
- Glenn Danzig's music incorporates influences from classical music. [1]
- Glenn Danzig has written a song for Roy Orbison. [1]
- Glenn Danzig is known for his tenor vocal range. [1]
- Glenn Danzig's vocal style has been linked to Howlin' Wolf. [1]
- Glenn Danzig cited Bill Medley as a vocal influence. [1]
- In 2023, Rolling Stone ranked Glenn Danzig at number 199 on its list of the 200 Greatest Singers of All Time. [1]
- Glenn Danzig was born on June 23, 1955. [2]
- Glenn Danzig was born in Lodi, New Jersey. [2, 3]
- Glenn Danzig disbanded the band Misfits in October 1983 due to personal and professional differences. [2]
- Glenn Danzig is a record producer. [3]
- Glenn Danzig owns the Evilive record label. [3]

Original question:
{query}

{background_texts}

Evaluated answer:
{answer}

Reasoning:{prompt_end}"""


class CoverageResult(TypedDict):
    """
    Stores the coverage results for a statement.
    """

    statement: str
    is_covered: bool
    context_ids: list[str]


class CoverageEvaluator:
    """
    Determines covered/uncovered contexts directly using an LLM.

    Used as a baseline for comprehensiveness evaluation.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-instruct",
        version: Literal["base", "few-shot"] = "few-shot",
        inject_prompt_template: bool = True,
    ):
        """
        Initializes the OutputGenerator.

        Args:
            model: str
                The LLM model to use for generating the output.
            version: str
                The prompt version to use. Either "base" or "few-shot".
            inject_prompt_template: bool
                Whether to inject prompt template tags to the prompt.
                Defaults to True.
        """
        rits_model_info = RITS_MODELS[model]
        if inject_prompt_template:
            self.prompt_begin = rits_model_info.get("prompt_begin")
            self.prompt_end = rits_model_info.get("prompt_end")
        else:
            self.prompt_begin = ""
            self.prompt_end = ""
        self.version = version
        self.llm_handler = LLMHandler(model, RITS=True)

        print(f"[CoverageEvaluator] Using LLM on RITS: {model}")

    def make_prompt(self, query: str, background_texts: list[str], answer: str) -> str:
        """
        Constructs a prompt for determining the covered/uncovered contexts.

        Args:
            query: str
                The user query for which the answer was generated.
            background_texts: list[str]
                The list of contexts to be used for evaluating comprehensiveness.
            answer: str
                The answer for which comprehensiveness should be evaluated.

        Returns:
            str:
                The constructed prompt.
        """
        background_texts_str = "\n\n".join(
            [
                f"Background text #{i + 1}:\n{text}"
                for i, text in enumerate(background_texts)
            ]
        )
        if self.version == "base":
            template = COVERAGE_EVALUATOR_BASE_PROMPT
        else:
            template = COVERAGE_EVALUATOR_PROMPT
        return template.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            background_texts=background_texts_str,
            answer=answer,
        )

    def run(
        self, query: str, background_texts: dict[str, str], answer: str
    ) -> tuple[list[CoverageResult], list[CoverageResult]]:
        """
        Determines the context statements covered and uncovered in the given answer.

        Args:
            query: str
                The user query for which the answer was generated.
            background_texts: dict[str, str]
                A dictionary mapping context IDs to their texts.
            answer: str
                The answer for which comprehensiveness should be evaluated.

        Returns:
            tuple[list[CoverageResult], list[CoverageResult]]:
                The list of covered and uncovered statements, respectively,
                wrapped in CoverageResult objects including additional metadata
                (such as the IDs of the corresponding contexts).
        """
        prompt = self.make_prompt(query, list(background_texts.values()), answer)
        print("[CoverageEvaluator] Prompt created: 1")

        covered_results_text = ""
        uncovered_results_text = ""
        try:
            for attempt in Retrying(
                stop=stop_after_attempt(3),
                wait=wait_random_exponential(multiplier=10, max=30),
            ):
                with attempt:
                    response = self.llm_handler.completion(
                        prompt, max_tokens=16384, seed=42, expect_content=True
                    )
                    response_text: str = response.choices[0].message.content  # type: ignore
                    print(response_text)
                    final_response_text = regex.split(
                        r"(?i)[\[*]+Covered statements[:\]*]+\s*(?=-)", response_text
                    )[-1]
                    covered_results_text, uncovered_results_text = regex.split(
                        r"(?i)[\[*]+Uncovered statements[:\]*]+\s*(?=-)",
                        final_response_text,
                        maxsplit=1,
                    )
        except RetryError:
            # We failed to extract
            print("WARNING: Failed to extract covered/uncovered contexts!")
            traceback.print_exc()
            return [], []

        result_pattern = r"(?m)^\s*-\s*(.+?)\s*(\[[0-9 ,]+?\])\s*$"
        all_results: list[list[CoverageResult]] = []
        for i, result_text in enumerate([covered_results_text, uncovered_results_text]):
            result_items = regex.findall(result_pattern, result_text)
            results = []
            for statement, context_idxs in result_items:
                # Parse numerical context indices
                try:
                    context_idxs_list: list[int] = ast.literal_eval(context_idxs)
                    if (
                        not isinstance(context_idxs_list, list)
                        or not context_idxs_list
                        or any(not isinstance(e, int) for e in context_idxs_list)
                    ):
                        raise ValueError(f"Invalid context indices: {context_idxs}")
                    context_idxs_list = [
                        i for i in context_idxs_list if i - 1 < len(background_texts)
                    ]
                    if not context_idxs_list:
                        raise ValueError(
                            f"Invalid values in context indices: {context_idxs}"
                        )
                except Exception:
                    print(
                        f"WARNING: Failed to parse context indices {context_idxs} due to the following exception:"
                    )
                    traceback.print_exc()
                    continue

                # Translate numerical indices to string IDs
                all_context_ids = list(background_texts.keys())
                context_ids = [all_context_ids[i - 1] for i in context_idxs_list]

                results.append(
                    CoverageResult(
                        statement=statement, is_covered=i == 0, context_ids=context_ids
                    )
                )

            all_results.append(results)

        covered_results, uncovered_results = all_results
        return covered_results, uncovered_results
