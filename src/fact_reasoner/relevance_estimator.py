import math

from itertools import batched
from litellm.types.utils import ModelResponse
from tqdm.auto import tqdm

from src.fact_reasoner.fact_utils import extract_weighted_scores
from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS

RELEVANCE_ESTIMATOR_PROMPT = """{prompt_start}You are an AI assistant evaluating the relevance of multiple pieces of background information to a specific user query.

For each background information item, you should assign a relevance score based on how important the information is for answering the original user query. Use the following relevance scale:
* [Relevance: 1] — The fact is entirely unrelated to the user query.
* [Relevance: 2] — The fact is topically related to the query but does not contribute meaningfully to answering it.
* [Relevance: 3] — The fact could be included in a comprehensive or extended answer, but is not necessary for a concise or focused response.
* [Relevance: 4] — The fact would typically be included in a good answer to the query, but its omission would not make the answer incorrect.
* [Relevance: 5] — The fact is essential to answering the query. Any valid response must include this information.

You should output the relevance classifications in the form of a list, with each item starting with "* ", followed by the item text and the relevance score in the format [Relevance: <relevance score>]. Do not include any other explanations or comments. Refer to the following examples to understand the task and the output format.

Example 1:

User query:
Tell me about Glenn Danzig.

Background information items:
* Glenn Danzig was born on June 23, 1995.
* Glenn Danzig is an American.
* Glenn Danzig is a singer.
* Glenn Danzig is a songwriter.
* Glenn Danzig is a musician.
* Glenn Danzig is a record producer.
* Glenn Danzig is the founder of the rock band Misfits.
* Glenn Danzig is the founder of the rock band Samhain.
* Glenn Danzig is the founder of the rock band Danzig.
* Glenn Danzig owns the Evilive record label.
* Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company.

Relevance classifications:
* Glenn Danzig was born on June 23, 1995. [Relevance: 4]
* Glenn Danzig is an American. [Relevance: 4]
* Glenn Danzig is a singer. [Relevance: 5]
* Glenn Danzig is a songwriter. [Relevance: 5]
* Glenn Danzig is a musician. [Relevance: 5]
* Glenn Danzig is a record producer. [Relevance: 5]
* Glenn Danzig is the founder of the rock band Misfits. [Relevance: 5]
* Glenn Danzig is the founder of the rock band Samhain. [Relevance: 5]
* Glenn Danzig is the founder of the rock band Danzig. [Relevance: 5]
* Glenn Danzig owns the Evilive record label. [Relevance: 4]
* Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company. [Relevance: 3]

Example 2:

User query:
What is the maximum range of Airbus A380?

Background information items:
* Airbus A380 is a very large wide-body airliner.
* Airbus A380 was developed and produced by Airbus until 2021.
* Airbus A380 is the world's largest passenger airliner.
* Airbus A380 is the only full-length double-deck jet airliner.
* The Airbus A380 is a quadjet.
* The Airbus A380 is powered by Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans.
* The Airbus A380 has a range of 8,000 nmi (14,800 km; 9,200 mi). 
* As of December 2021, the global A380 fleet had completed more than 800,000 flights.
* As of December 2021, Airbus A380 experienced no hull losses.

Relevance classifications:
* Airbus A380 is a very large wide-body airliner. [Relevance: 2]
* Airbus A380 was developed and produced by Airbus until 2021. [Relevance: 2]
* Airbus A380 is the world's largest passenger airliner. [Relevance: 2]
* Airbus A380 is the only full-length double-deck jet airliner. [Relevance: 2]
* The Airbus A380 is a quadjet. [Relevance: 2]
* The Airbus A380 is powered by Engine Alliance GP7200 or Rolls-Royce Trent 900 turbofans. [Relevance: 2]
* The Airbus A380 has a range of 8,000 nmi (14,800 km; 9,200 mi). [Relevance: 5]
* As of December 2021, the global A380 fleet had completed more than 800,000 flights. [Relevance: 2]
* As of December 2021, Airbus A380 experienced no hull losses. [Relevance: 2]

Now, please determine the relevance of the following background items to the provided user query.

User query:
{query}

Background information items:
{background_items}

Relevance classifications:{prompt_end}"""

RELEVANCE_PATTERN = r"\*\s+(.+?)\s+\[Relevance:\s+([0-9]+)\]"


class RelevanceEstimator:
    """
    Estimates the relevance of statements to a specific user query.
    """

    def __init__(self, model: str = "llama-3.3-70b-instruct"):
        """
        Initializes the RelevanceEstimator.

        Args:
            model: str
                The LLM model to use for determining statement relevance.
            use_logprobs: bool
                Whether to extract probabilistic relevance scores using
                log-probabilities. Defaults to True.
        """
        rits_model_info = RITS_MODELS[model]
        self.prompt_begin = rits_model_info.get("prompt_begin")
        self.prompt_end = rits_model_info.get("prompt_end")
        self.llm_handler = LLMHandler(model, RITS=True)

        print(f"[RelevanceEstimator] Using LLM on RITS: {model}")

    def make_prompt(self, query: str, background_items: list[str]) -> str:
        """
        Constructs a prompt for estimating statement relevance.

        Args:
            query: str
                The original user query with respect to which the
                relevance should be estimated.
            background_items: list[str]
                The list of statements for which to estimate relevance.

        Returns:
            str:
                The constructed prompt.
        """
        background_items_list = "\n".join(
            [f"{i + 1}. {item}" for i, item in enumerate(background_items)]
        )
        return RELEVANCE_ESTIMATOR_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            background_items=background_items_list,
        )

    def extract_result(self, response: ModelResponse) -> list[tuple[str, float]]:
        weighted_scores = extract_weighted_scores(
            response, r"\*\s+(.+?)\s+\[Relevance:\s+([0-9]+)\]"
        )
        return weighted_scores

    def run(
        self,
        query: str,
        background_items: list[str],
    ) -> list[tuple[str, float]]:
        """
        Estimates the relevance of statements to a specific user query.

        Args:
            query: str
                The original user query with respect to which the
                relevance should be estimated.
            background_items: list[str]
                The list of statements for which to estimate relevance.

        Returns:
            list[tuple[str, float]]:
                The list of extracted pairs of atoms and their respective
                relevance scores.
        """
        prompt = self.make_prompt(query, background_items)
        print("[RelevanceEstimator] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt,
            max_tokens=1024,
            seed=42,
            logprobs=True,
            top_logprobs=20,
            expect_content=True,
        )
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {response}")
        return extract_weighted_scores(response, RELEVANCE_PATTERN)

    def runall(
        self,
        queries: list[str],
        background_items: list[list[str]],
        batch_size: int = 16,
    ) -> list[list[tuple[str, float]]]:
        """
        Estimates the relevance of statements to specific user queries,
        using batched LLM calls.

        Args:
            queries: list[str]
                The original user queries with respect to which the
                relevance should be estimated.
            background_items: list[list[str]]
                The list of statements for which to estimate relevance,
                for each query.
            batch_size: int
                The maximum amount of parallel model calls.

        Returns:
            list[list[tuple[bool, float]]]:
                The nested list of extracted pairs of atoms and their
                respective relevance scores, for each set of statements.
        """
        prompts = [self.make_prompt(q, bi) for q, bi in zip(queries, background_items)]
        print(f"[RelevanceEstimator]: Prompts created: {len(prompts)}")

        all_results: list[list[tuple[str, float]]] = []
        for prompt_batch in tqdm(
            batched(prompts, n=batch_size),
            total=math.ceil(len(prompts) / batch_size),
        ):
            responses = self.llm_handler.batch_completion(
                prompt_batch,
                max_tokens=9000,
                seed=42,
                logprobs=True,
                top_logprobs=20,
                expect_content=True,
            )
            for response in responses:
                if not isinstance(response, ModelResponse):
                    raise ValueError(f"Unexpected type for model response: {response}")
                results = extract_weighted_scores(response, RELEVANCE_PATTERN)
                all_results.append(results)

        return all_results
