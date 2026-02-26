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

import mellea.stdlib.functional as mfuncs

from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

# Local imports
from fact_reasoner.utils import validate_markdown_code_block, strip_code_fences

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
   
2. Format Final Query:
   Present your query wrapped between Markdown code fences:
   ```
    <your generated query here>
   ```

Use the following examples to learn the task better.

Example 1:
STATEMENT: The Great Wall of China is visible from space
OUTPUT:
```
"The Great Wall of China is visible from space" fact check myth
```

Example 2:
STATEMENT: Apple will release a foldable iPhone in 2026
OUTPUT:
```
"Apple will release a foldable iPhone in 2026" rumor OR announcement
```

Example 3:
STATEMENT: Quantum computers can break RSA encryption easily
OUTPUT:
```
"Quantum computers can break RSA encryption easily" fact check cryptography experts
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
        print(f"[QueryBuilder] Using Mellea backend: {self.backend.model_id}")

        # Disable Mellea logging
        FancyLogger.get_logger().setLevel(FancyLogger.ERROR)


    def run(self, text: str) -> str:
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
                    "The output must be wrapped with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_markdown_code_block(s)
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
            return cleaned
        else:
            return text # the original text
                        
