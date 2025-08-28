from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS


OUTPUT_GENERATOR_PROMPT = """{prompt_start}You are an AI assistant specialized in answering user queries based on a set of background texts.

Your task is to generate an answer to the user query using only the information found in the background texts. Your response must:
* Be strictly relevant to the query, avoiding any unrelated content.
* Be comprehensive and include important contextual details such as reasons, arguments, and justifications.
* Represent all differing viewpoints found in the background texts, even if they contradict each other or your own knowledge.
* Avoid referencing ID numbers or metadata of the background texts.
* Be entirely based on the information in the background texts.

Respond only with the answer to the user query, without any other comments or explanations. If the background texts contain no relevant information, reply "Sorry, I don't have any information relevant to the given query."

User query:
{query}

{background_texts}

Answer:{prompt_end}"""


class OutputGenerator:
    """
    Generates an answer to the given user query based on the provided background texts.
    """

    def __init__(
        self, model: str = "llama-3.3-70b-instruct", inject_prompt_template: bool = True
    ):
        """
        Initializes the OutputGenerator.

        Args:
            model: str
                The LLM model to use for generating the output.
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

        print(f"[OutputGenerator] Using LLM on RITS: {model}")

    def make_prompt(self, query: str, background_texts: list[str]) -> str:
        """
        Constructs a prompt for generating an answer to the given query.

        Args:
            query: str
                The user query for which the answer should be generated.
            background_texts: list[str]
                The list of contexts to use for answering the user query.

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
        return OUTPUT_GENERATOR_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            query=query,
            background_texts=background_texts_str,
        )

    def run(self, query: str, background_texts: list[str]) -> str:
        """
        Generates an answer to the given user query based on the provided
        background texts.

        Args:
            query: str
                The user query for which the answer should be generated.
            background_texts: list[str]
                The list of contexts to use for answering the user query.

        Returns:
            str:
                The generated answer.
        """
        prompt = self.make_prompt(query, background_texts)
        print("[OutputGenerator] Prompt created: 1")
        response = self.llm_handler.completion(
            prompt, max_tokens=9000, seed=42, expect_content=True
        )
        return response.choices[0].message.content  # type: ignore
