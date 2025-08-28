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

# Query builder for atoms to retrieve results from Google and/or Wikipedia

from tqdm import tqdm
from typing import Dict, List

from src.fact_reasoner.utils import extract_last_square_brackets
from src.fact_reasoner.llm_handler import LLMHandler

# Single turn version
QUERY_BUILDER_PROMPT_V1 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
Your task is to generate a Google Search query about a given STATEMENT. \
Optionally, you are also given a list of previous queries and results called KNOWLEDGE. \
Your goal is to generate a high quality query that is most likely to retrieve the relevant information about the STATEMENT.

QUERY CONSTRUCTION CRITERIA: a well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Seek new information not present in the current KNOWLEDGE.
  - Balance specificity for targeted results with breadth to avoid missing critical information.

Process:
1. Construct a Useful Google Search Query: 
  - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
  - Prioritize natural language queries that a typical user might enter.
  - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.

2. Provide Query Rationale (2-3 sentences): 
  Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query: 
  Finally, present your query enclosed in square brackets, like [QUERY].

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

# Multi-turn version (experimental)
QUERY_BUILDER_PROMPT_V2 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
You are engaged in a multi-round process to refine Google Search queries about a given STATEMENT. \
Each round builds upon KNOWLEDGE (a list of previous queries and results, starting empty in round 1). \
Your goal is to improve query quality and relevance over successive rounds.

QUERY CONSTRUCTION CRITERIA: a well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Seek new information not present in the current KNOWLEDGE.
  - Balance specificity for targeted results with breadth to avoid missing critical information.
  - In rounds 2+, leverage insights from earlier queries and outcomes.

Process:
1. Construct a Useful Google Search Query: 
  - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
  - Prioritize natural language queries that a typical user might enter.
  - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.

2. Provide Query Rationale (2-3 sentences): 
  Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query: 
  Present your query enclosed in square brackets, like [QUERY].

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""

class QueryBuilder:
    """
    The QueryBuilder uses an LLM to generate a query string. The query is then
    used to retrieve results from Google Search, Wikipedia, ChromaDB.
    """

    def __init__(self, model: str, prompt_version: str = "v1", use_rits: bool = True):
        """
        Initialize the QueryBuilder

        Args:
            model: str
                The name of the LLM used for query generation.
            rits: bool
                A boolean flag indicating a remote RITS model or a local model.
        """

        self.model = model
        self.prompt_version = prompt_version
        self.use_rits = use_rits
        self.llm_handler = LLMHandler(model, RITS=use_rits)
    
    def make_prompt(self, statement: str, knowledge: str = "") -> str:
        """
        Creat the prompt for a given atom and previous retrieved results.
        """

        if self.prompt_version == "v1":
            prompt = QUERY_BUILDER_PROMPT_V1.format(
                _STATEMENT_PLACEHOLDER=statement,
                _KNOWLEDGE_PLACEHOLDER=knowledge,
                _PROMPT_BEGIN_PLACEHOLDER=self.llm_handler.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.llm_handler.prompt_end, 
            )
        elif self.prompt_version == "v2":
            prompt = QUERY_BUILDER_PROMPT_V2.format(
                _STATEMENT_PLACEHOLDER=statement,
                _KNOWLEDGE_PLACEHOLDER=knowledge,
                _PROMPT_BEGIN_PLACEHOLDER=self.llm_handler.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.llm_handler.prompt_end, 
            )

        return prompt

    def run(self, statement: str, knowledge: str = "") -> Dict[str, str]:
        """
        Generate the query for a given statement and knowledge (if any).

        Args:
            statement: str
                The input statement (e.g., an atomic unit).
            knowledge: str
                The input knowledge, i.e., a list of previuos queries and results.
        
        Return:
            A dict containg the `query` and the verbose `response`.
        """

        prompt = self.make_prompt(statement, knowledge)
        response = self.llm_handler.completion(prompt)
        generated_text = response.choices[0].message.content
        query = extract_last_square_brackets(generated_text)

        assert query is not None and len(query) > 0, f"Could not generate the `query`."
        return dict(query=query, response=generated_text)

    def runall(self, statements: List[str], knowledges: List[str]) -> List[Dict[str, str]]:
        """
        Generate the queries for a list of statements and knowledges (if any).

        Args:
            statements: List[str]
                The list of input statements (e.g., atomic units).
            knowledges: List[str]
                The list of input knowledges (previuos queries and results).
        
        Return:
            A list of dicts containg the `query` and the verbose `response`.
        """

        # Safety checks
        assert len(statements) == len(knowledges), \
            f"Length of `statements` must be equal to the length of `knowledges`."
        
        prompts = [self.make_prompt(statements[i], knowledges[i]) for i in range(len(statements))]
        generated_texts = []
        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(prompts)
            ),
            total=len(prompts),
            desc="Query Builder",
            unit="prompts",
            ):
                generated_texts.append(response.choices[0].message.content)

        result = []
        for generated_text in generated_texts:
            query = extract_last_square_brackets(generated_text)
            assert query is not None and len(query) > 0, f"Could not generate the `query`."
            result.append(dict(query=query, response=generated_text))

        return result

if __name__ == "__main__":

    model = "mixtral-8x22b-instruct"
    prompt_version = "v1"
    use_rits = True

    qbuilder = QueryBuilder(model=model, prompt_version=prompt_version, use_rits=use_rits)

    # Process a single atom (no knowledge)
    # atom = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    atom = "You'd have to yell if your friend is outside the same location"

    result = qbuilder.run(atom)
    query = result["query"]
    response = result["response"]
    print(f"Atom: {atom}")
    print(f"Response: {response}")
    print(f"Query: {query}")

    # Process multiple atoms (no knowledge)
    # print(f"Generating queries for multiple atoms...")
    # print("-------"*10)
    # atoms = [
    #     "Lanny Flaherty was born in Pensacola, Florida", 
    #     "Vitamin C is taken from oranges.",
    #     "Ko Itakura is a professional football player.",
    #     "Rin Iwanaga is a fictional character.",
    #     "Joeri Adams has competed for Telenet-Fidea.",
    #     "Regina Mart\u00ednez P\u00e9rez was born on October 28, 1963.",
    #     "Don Featherstone was born in Worcester.",
    #     "Julia Faye began her career as a child actress.",
    #     "Craig Morton attended the University of California.",
    #     "Alexandre Guilmant began studying music with his father.",
    # ]
    # knwoledges = [""] * 10
    # results = qbuilder.runall(atoms, knwoledges)
    # for i, result in enumerate(results):
    #     query = result["query"]
    #     atom = atoms[i]
    #     response = result["response"]
    #     print(f"Atom: {atom}")
    #     print(f"Response: {response}")
    #     print(f"Query: {query}")
    #     print("-------"*10)


    print("Done.")
