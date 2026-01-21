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

import json
import asyncio
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext
from mellea.stdlib.requirement import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.utils import validate_json_code_block, strip_code_fences, escape_quotes

INSTRUCTION_QUERY_BUILDER = """
Instructions:
Your task is to generate a Google Search query about a given STATEMENT. Your goal is to create a high-quality query that is most likely to retrieve relevant information about the STATEMENT.

QUERY CONSTRUCTION CRITERIA:
A well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Balance specificity for targeted results with breadth to avoid missing critical information.

Process:
1. Construct a Useful Google Search Query:
   - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
   - Prioritize natural language queries that a typical user might enter.
   - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.
   

2. Provide Query Rationale (2-3 sentences):
   Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query:
   Present your query and rationale in the following JSON format:
   ```json
   {
       "query": "<your generated query here>",
       "rationale": "<your rationale here>"
   }
   ```

Use the following examples to learn the task better.

Example 1:
STATEMENT: The Great Wall of China is visible from space
OUTPUT:
```json
{
    "query": "\"The Great Wall of China is visible from space\" fact check myth",
    "rationale": "This query uses quotation marks to ensure the exact statement is searched and adds 'fact check' and 'myth' to retrieve authoritative sources that address the claim's accuracy."}
```

Example 2:
STATEMENT: Apple will release a foldable iPhone in 2026
OUTPUT:
```json
{
    "query": "\"Apple will release a foldable iPhone in 2026\" rumor OR announcement",
    "rationale": "Including the statement in quotes ensures precision, while adding keywords like 'rumor' and 'announcement' helps capture both official sources and credible tech news discussing the claim."
}
```

Example 3:
STATEMENT: Quantum computers can break RSA encryption easily
OUTPUT:
```json
{
    "query": "\"Quantum computers can break RSA encryption easily\" fact check cryptography experts",
    "rationale": "The query uses quotation marks for accuracy and adds 'fact check' and 'cryptography experts' to find authoritative sources that evaluate the feasibility of this claim."
}
```

Your task:

STATEMENT: {{statement_text}}
OUTPUT:
"""

class QueryBuilder:
    """
    The QueryBuilder uses an LLM to generate a query string. The query is then
    used to retrieve results from Google Search, Wikipedia, ChromaDB.
    """

    def __init__(
        self, 
        backend: Backend
    ):
        """
        Initialize the QueryBuilder.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interaction.
        """

        # Safety checks        
        if backend is None:
            raise ValueError("Mellea backend is None. Please provide a valid Mellea backend.")

        # Initialize the extractor
        self.backend = backend
        
        # Print info
        print(f"[Atomizer] Using Mellea backend: {self.backend.model_id}")

    def run(self, text: str) -> Dict[str, Any]:
        """
        Build a Google search query for the given text.
        
        Args:
            text: str
                The text for which to build the Google search query.
        Returns:
            dict: A dictionary containing the query string.
        """
        
        # Perform the instruction with validation
        output = mfuncs.instruct(
            INSTRUCTION_QUERY_BUILDER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s, required_keys=["query", "rationale"])
                    ),
                )
            ],
            user_variables={"statement_text": text},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            cleaned = escape_quotes(cleaned)
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure
                        
    async def arun(self, text: str) -> Dict[str, Any]:
        """
        Build a Google search query for the given text.
        
        Args:
            text: str
                The text for which to build the Google search query.
        Returns:
            dict: A dictionary containing the query text.
        """
        
        # Perform the instruction with validation
        output = await mfuncs.ainstruct(
            INSTRUCTION_QUERY_BUILDER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s, required_keys=["query", "rationale"])
                    ),
                )
            ],
            user_variables={"statement_text": text},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure

if __name__ == "__main__":

    use_async = False

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500},
    )

    # Create the query builder
    qb = QueryBuilder(backend)

    # Process a single atom (no knowledge)        
    text = "The Apollo 14 mission to the Moon took place on January 31, 1971."
    # text = "You'd have to yell if your friend is outside the same location"

    if not use_async:
        result = qb.run(text)
        print(f"Query builder result: {result}")
    else:
        result = asyncio.run(qb.arun(text))
        print(f"Async query builder result: {result}")

    # Print the query
    print(f"Initial Text: {text}")
    print(f"Query: {result.get('query', '')}")

    print("Done.")
