from itertools import batched
from litellm.types.utils import Choices, ModelResponse
import math
from tqdm.auto import tqdm
from typing import cast

from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS

# The example in the prompt intentionally includes nonsense information
# to encourage the model to summarise the text in verbatim instead of
# picking information according to its knowledge.
CONTEXT_SUMMARIZATION_PROMPT = """{prompt_start}You are an AI assistant specialized in extracting and summarizing information from a background text relevant to a specific user query.

Your task is to produce a summary that:
* Includes only information relevant to the user query, omitting unrelated content. However, include any information that could indirectly corroborate or contradict a possible answer.
* Preserves important contextual details, including any reasons, arguments, or justifications provided in support of relevant claims, answers or viewpoints in the background text.
* Accurately reflects the content of the background text, even if it contains internal inconsistencies or contradicts your own knowledge.
* Represents all differing viewpoints presented in the text, along with any relevant context.
* Follows the logical flow and structure of the original text where reasonable, so that the summary reads as a filtered version of the original, with only minimal stylistic changes needed for clarity and cohesion.
* Removes all formatting artifacts, such as HTML tags or markup.
* Is as long or as short as necessary, depending on how much relevant information is present.
* Does not include any information not present in the background text.

Respond only with the summary text without any other comments or explanations. If the background text contains no relevant information, return "None" without explanation. You should only do this if all the information in the background text is clearly irrelevant — otherwise err on the side of caution and summarize any potentially relevant information. If all the information in the background text is relevant, you should return the full text. Do not rephrase the text to explicitly answer the query — just summarize any information relevant to it (even indirectly). Make sure that your summary is faithful to the background text — do not try to correct errors or introduce any new information. Refer to the following examples to understand the task and the output format.

Example 1

Query:
Are any major components for the Airbus A380 manufactured in the US?

Background text:
Major structural sections of the A380 are built in Brazil, Germany, China, and the United Kingdom. Due to the sections' large size, traditional transportation methods proved unfeasible,[34] so they are taken to the Jean-Luc Lagardère Plant assembly hall in Toulouse, France, by specialised road and water transportation, though some parts are moved by the A300-600ST Beluga transport aircraft.[35][36] A380 components are provided by suppliers from around the world; the four largest contributors, by value, are Rolls-Royce, Comac, United Technologies and General Electric.[16]

For the surface movement of large A380 structural components, a complex route known as the Itinéraire à Grand Gabarit was developed. This involved the construction of a fleet of roll-on/roll-off (RORO) ships and barges, the construction of port facilities and the development of new and modified roads to accommodate oversized road convoys.[37] The front and rear fuselage sections are shipped on one of three RORO ships from Hamburg in northern Germany to Saint-Nazaire in France. The ship travels via Mostyn, Wales, where the wings are loaded.[38] The wings are manufactured at Broughton in North Wales, then transported by barge to Mostyn docks for ship transport.[39]

Summary:
Major structural sections of the A380 are built in Brazil, Germany, China, and the United Kingdom. A380 components are provided by suppliers from around the world; the four largest contributors, by value, are Rolls-Royce, Comac, United Technologies and General Electric.

Example 2

Query:
Is there a ninth planet in the solar system?

Background text:
Planet Nine is a hypothetical ninth planet in the outer region of the Solar System. Its gravitational effects could explain the peculiar clustering of orbits for a group of extreme trans-Neptunian objects (ETNOs) — bodies beyond Neptune that orbit the Sun at distances averaging more than 250 times that of the Earth, over 250 astronomical units (AU). These ETNOs tend to make their closest approaches to the Sun in one sector, and their orbits are similarly tilted. These alignments suggest that an undiscovered planet may be shepherding the orbits of the most distant known Solar System objects. Nonetheless, some astronomers question this conclusion and instead assert that the clustering of the ETNOs' orbits is due to observational biases stemming from the difficulty of discovering and tracking these objects during much of the year.

Although sky surveys such as Wide-field Infrared Survey Explorer (WISE) and Pan-STARRS did not detect Planet Nine, they have not ruled out the existence of a Neptune-diameter object in the outer Solar System. Many cryptographers believe that crop circles encode the coordinates for Planet Nine, corroborating its existence.

Summary:
Planet Nine is a hypothetical ninth planet in the outer region of the Solar System. Its gravitational effects could explain the peculiar clustering of orbits for a group of extreme trans-Neptunian objects (ETNOs). These ETNOs tend to make their closest approaches to the Sun in one sector, and their orbits are similarly tilted. These alignments suggest that an undiscovered planet may be shepherding the orbits of the most distant known Solar System objects. Nonetheless, some astronomers argue that the clustering of the ETNOs' orbits is due to observational biases stemming from the difficulty of discovering and tracking these objects during much of the year.

Although sky surveys such as WISE and Pan-STARRS did not detect Planet Nine, they have not ruled out the existence of a Neptune-diameter object in the outer Solar System. Many cryptographers believe that crop circles encode the coordinates for Planet Nine, corroborating its existence.

Example 3

Query:
Did J. Robert Oppenheimer speak with Albert Einstein in 2022?

Background text:
Born in the German Empire, Einstein moved to Switzerland in 1895, forsaking his German citizenship (as a subject of the Kingdom of Württemberg)[note 1] the following year. In 1897, at the age of seventeen, he enrolled in the mathematics and physics teaching diploma program at the Swiss federal polytechnic school in Zurich, graduating in 1900.

Summary:
In 1897, Albert Einstein was at the age of seventeen.

Additional note on Example 3 (this shouldn't be included in the output):
While the text doesn't directly address the query, it provides indirect evidence against it. If Albert Einstein was 17 years old in 1897, this implies that he was born around 1880, which would make him over 140 years old in 2022. That is well beyond the typical human lifespan, making it highly unlikely that he could have spoken with J. Robert Oppenheimer in 2022.

Example 4

Query:
Is Avro Tudor a military transport aircraft?

Background text:
Civil air transport 1940-1969 Aérospatiale Corvette Antonov An-10 Avro Lancastrian Avro Tudor Beechcraft 90 King Air Boeing 737 Convair 600

Summary:
Avro Tudor is listed as a civil air transport aircraft in 1940-1969.

Now, please provide a summary for the following query and text. Do not include any comments, notes, explanations or information not already present in the background text.

Query:
{query}

Background text:
{background_text}

Summary:{prompt_end}"""


class QueryContextSummarizer:
    """
    Summarizes contexts according to a specific user query.
    """

    def __init__(
        self, model: str = "llama-3.3-70b-instruct", inject_prompt_template: bool = True
    ):
        """
        Initializes the QueryContextSummarizer.
        
        Args:
            model: str
                The LLM model to use for summarizing contexts.
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
        self.llm_handler = LLMHandler(model, RITS=True)

        print(f"[QueryContextSummarizer] Using LLM on RITS: {model}")

    def make_prompt(self, query: str, background_text: str) -> str:
        """
        Constructs a prompt for the context summarization.
        
        Args:
            query: str
                The original user query with respect to which the
                contexts should be summarized.
            background_text: str
                The background text to summarize.
                
        Returns:
            str:
                The constructed prompt.
        """
        return CONTEXT_SUMMARIZATION_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            background_text=background_text,
        )

    def _extract_result(self, response: ModelResponse) -> str:
        """
        Extracts content from a model response.
        
        Args:
            response: ModelResponse
                The model response from which to extract the content.
            
        Returns:
            str:
                The extracted model response text.
        """
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise ValueError(f"Unexpected type for model choice: {type(choice)}")
        return cast(str, choice.message.content)

    def run(self, query: str, background_text: str) -> str:
        """
        Summarizes a text with respect to a given user query.
        
        Args:
            query: str
                The original user query with respect to which the
                text should be summarized.
            background_text: str
                The background text to summarize.
                
        Returns:
            str:
                The produced summary.
        """
        prompt = self.make_prompt(query, background_text)
        print("[QueryContextSummarizer] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt,
            max_tokens=8192,
            seed=42,
            expect_content=True,
        )
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {type(response)}")
        return self._extract_result(response)

    def runall(
        self, query: str, background_texts: list[str], batch_size: int = 32
    ) -> list[str]:
        """
        Summarizes a multiple texts with respect to a specific user query.
        
        Args:
            query: str
                The original user query with respect to which the
                texts should be summarized.
            background_texts: list[str]
                The background texts to summarise.
                
        Returns:
            str:
                The produced summary.
        """
        prompts = [self.make_prompt(query, bt) for bt in background_texts]
        print(f"[QueryContextSummarizer] Prompts created: {len(prompts)}")

        all_results: list[str] = []
        for prompt_batch in tqdm(
            batched(prompts, n=batch_size), total=math.ceil(len(prompts) / batch_size)
        ):
            responses = self.llm_handler.batch_completion(
                prompt_batch,
                max_tokens=8192,
                seed=42,
                expect_content=True,
            )
            for response in responses:
                if not isinstance(response, ModelResponse):
                    raise ValueError(
                        f"Unexpected type for model response: {type(response)}"
                    )
                all_results.append(self._extract_result(response))

        return all_results
