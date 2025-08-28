from itertools import batched
from litellm.types.utils import Choices, ModelResponse
import math
from tqdm import tqdm

from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS

ATOM_REVISER_PROMPT = """{prompt_start}Your task is to revise the statements below to make them self-contained. Each of these statements has been extracted from a background text and should be considered in the context of this text.

You should adjust each statement to resolve any vague references, such as:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Follow the following steps for revising each statement.
1. If the statement contains vague references, minimally revise them with respect to the specific subjects they refer to in the background text.
2. Each statement should be minimally revised by only resolving vague references. No changes should be made to the content and no additional information should be added.
3. However, if there are any conjunctive statements, they should be decomposed into multiple atomic units (e.g., Democracies treat citizens as equals regardless of their race or religion. → Democracies treat citizens as equals regardless of their race., Democracies treat citizens as equals regardless of their religion.). Avoid adding duplicate statements.
4. Provide each self-contained statement on a separate line starting with "* ". Do not provide any additional explanations or comments.

Refer to the following examples to understand the task and the output format.

Example 1:

Background text:
Glenn Danzig (born June 23, 1955) is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.

Statements to be revised:
* Glenn Danzig was born on June 23, 1955.
* Glenn Danzig is an American.
* Glenn Danzig is a singer, songwriter, musician, and record producer.
* He is the founder of the rock bands Misfits, Samhain, and Danzig.
* He owns the record label.
* He owns Verotik.
* It is an adult-oriented comic book publishing company.

Revised statements:
* Glenn Danzig was born on June 23, 1955.
* Glenn Danzig is an American.
* Glenn Danzig is a singer.
* Glenn Danzig is a songwriter.
* Glenn Danzig is a musician.
* Glenn Danzig is a record producer.
* Glenn Danzig is the founder of the rock band Misfits.
* Glenn Danzig is the founder of the rock band Samhain.
* Glenn Danzig is the founder of the rock band Danzig.
* Glenn Danzig owns the Evilive record label.
* Glenn Danzig owns Verotik.
* Verotik is an adult-oriented comic book publishing company.

Example 2:

Background text:
With a total height of 829.8 m (2,722 ft, or just over half a mile) and a roof height (excluding the antenna, but includes a 242.6 m spire) of 828 m (2,717 ft), Burj Khalifa is the world's tallest structure.

Statements to be revised:
* The structure has a total height of 829.8 m (2,722 ft, or just over half a mile).
* The structure has a roof height (excluding the antenna, but includes a 242.6 m spire) of 828 m (2,717 ft).
* Burj Khalifa is the world's tallest structure.

Revised statements:
* Burj Khalifa has a total height of 829.8 m (2,722 ft, or just over half a mile).
* Burj Khalifa has a roof height (excluding the antenna, but includes a 242.6 m spire) of 828 m (2,717 ft).
* Burj Khalifa is the world's tallest structure.

Example 3:

Background text:
The Airbus A380 is a very large wide-body airliner, developed and produced by Airbus until 2021. It is the world's largest passenger airliner and the only full-length double-deck jet airliner. 

Statements to be revised:
* The Airbus A380 is a very large wide-body airliner.
* The aircraft was developed and produced by Airbus until 2021.
* It is the world's largest passenger airliner.
* It is the only full-length double-deck jet airliner.

Revised statements:
* The Airbus A380 is a very large wide-body airliner.
* The Airbus A380 was developed by Airbus.
* The Airbus A380 was produced by Airbus until 2021.
* The Airbus A380 is the world's largest passenger airliner.
* The Airbus A380 is the only full-length double-deck jet airliner.

Now, please revise the following statements.

Background text:
{background_text}

Statements to be revised:
{statement_items}

Revised statements:{prompt_end}"""


class BatchAtomReviser:
    """
    Revises batches of atoms to be self-contained. Differently from
    AtomReviser, this version considers multiple statements in the
    same model context.
    """

    def __init__(self, model: str = "llama-3.3-70b-instruct"):
        """
        Initializes the BatchAtomReviser.
        
        Args:
            model: str
                The name of the model to use for atom revision.
        """
        rits_model_info = RITS_MODELS[model]
        self.prompt_begin = rits_model_info.get("prompt_begin")
        self.prompt_end = rits_model_info.get("prompt_end")
        self.llm_handler = LLMHandler(model, RITS=True)

        print(f"[BatchAtomReviser] Using LLM on RITS: {model}")

    def make_prompt(self, background_text: str, statement_items: list[str]) -> str:
        """
        Constructs a prompt for revising a batch of atoms.
        
        Args:
            background_text: str
                The text with the necessary background information for
                revising the atoms.
            statement_items: list[str]
                The list of atomic statements to be revised.
        
        Returns:
            str:
                The constructed prompt."""
        statement_items_list = "\n".join(
            [f"* {item}" for i, item in enumerate(statement_items)]
        )
        return ATOM_REVISER_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            background_text=background_text,
            statement_items=statement_items_list,
        )

    def extract_results(self, response: ModelResponse) -> list[str]:
        """
        Extracts the revised atoms from a model response.
        
        Args:
            response: ModelResponse
                The model response from which the revised atoms should be
                extracted
        
        Returns:
            list[str]:
                The list of revised atoms.
        """
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            return []
        text_response = choice.message.content
        if text_response is None:
            return []

        revised_statements = text_response.split("\n")
        revised_statements = [
            statement.replace("*", "").strip()
            for statement in revised_statements
            if "*" in statement
        ]
        return list(dict.fromkeys(revised_statements))

    def run(self, background_text: str, statement_items: list[str]) -> list[str]:
        """
        Revises a set of atoms to be self-contained, using the provided
        background text.
        
        Args:
            background_text: str
                The text with the necessary background information for
                revising the atoms.
            statement_items: list[str]
                The list of atomic statements to be revised.
        
        Returns:
            list[str]:
                The list of revised atoms.
        """
        prompt = self.make_prompt(background_text, statement_items)
        print("[BatchAtomReviser] Prompt created: 1")
        response = self.llm_handler.completion(prompt, max_tokens=9000, seed=42, expect_content=True)
        if not isinstance(response, ModelResponse):
            raise ValueError(f"Unexpected type for model response: {response}")
        return self.extract_results(response)

    def runall(
        self,
        background_texts: list[str],
        statement_items: list[list[str]],
        batch_size: int = 16,
    ) -> list[list[str]]:
        """
        Revises several sets of atoms to be self-contained, using the
        provided background texts. Processes the sets of atoms in
        parallel.
        
        Args:
            background_texts: list[str]
                The list of texts providing the necessary background
                information for revising the atoms.
            statement_items: list[list[str]]
                The nested list of atomic statements to be revised
                for each background text.
            batch_size: int
                The maximum number of parallel model calls.
        
        Returns:
            list[list[str]]:
                The list of revised atoms for each background text.
        """
        prompts = [
            self.make_prompt(q, bi) for q, bi in zip(background_texts, statement_items)
        ]
        print(f"[BatchAtomReviser] Prompts created: {len(prompts)}")

        all_results: list[list[str]] = []
        for prompt_batch in tqdm(
            batched(prompts, n=batch_size), total=math.ceil(len(prompts) / batch_size)
        ):
            responses = self.llm_handler.batch_completion(
                prompt_batch, max_tokens=9000, seed=42, expect_content=True,
            )
            for response in responses:
                if not isinstance(response, ModelResponse):
                    raise ValueError(f"Unexpected type for model response: {response}")
                results = self.extract_results(response)
                all_results.append(results)

        return all_results
