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

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from .retriever import ContextRetriever
from .summarizer import ContextSummarizer
from .utils import Atom, Context


class ContextRetrieverFast:
    """
    Parallel context retriever that wraps a ContextRetriever and dispatches
    retrieval tasks across a thread pool.
    """

    def __init__(
        self,
        context_retriever: ContextRetriever,
        context_summarizer: Optional[ContextSummarizer] = None,
        num_workers: int = 4,
    ):
        self.context_retriever = context_retriever
        self.context_summarizer = context_summarizer
        self.num_workers = num_workers

    def _retrieve_for_item(
        self,
        text: str,
        atom: Optional[Atom] = None,
        id_prefix: str = "c",
    ) -> List[Context]:
        """Worker function: retrieve contexts (and optionally summarize) for one item."""
        retrieved = self.context_retriever.query(text=text)

        contexts = []
        for j, ctx in enumerate(retrieved):
            context = Context(
                id=f"{id_prefix}_{j}",
                atom=atom,
                text=ctx["text"],
                title=ctx["title"],
                link=ctx["link"],
                snippet=ctx["snippet"],
            )
            contexts.append(context)

        # Summarize in the same worker thread if summarizer is provided
        if self.context_summarizer is not None and atom is not None and len(contexts) > 0:
            results = asyncio.run(
                self.context_summarizer.run_batch(
                    [c.get_text() for c in contexts], atom.text
                )
            )
            for context, result in zip(contexts, results):
                if result["summary"]:
                    context.set_synthetic_summary(result["summary"])
                    context.set_probability(result["probability"] * context.get_probability())
            # Filter out irrelevant contexts (empty summary means not relevant)
            contexts = [c for c in contexts if c.get_summary()]

        return contexts

    def retrieve_all(
        self,
        atoms: Dict[str, Atom],
        query: Optional[str] = None,
    ) -> Dict[str, Context]:
        """Retrieve contexts for all atoms (and optionally the query) in parallel."""
        all_contexts: Dict[str, Context] = {}
        futures = {}

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit one task per atom
            for aid, atom in atoms.items():
                future = executor.submit(
                    self._retrieve_for_item, atom.text, atom, f"c_{aid}"
                )
                futures[future] = (aid, atom)

            # Submit query task
            if query:
                future = executor.submit(
                    self._retrieve_for_item, query, None, "c_q"
                )
                futures[future] = ("query", None)

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Retrieving contexts",
            ):
                aid, atom = futures[future]
                contexts = future.result()
                for ctx in contexts:
                    all_contexts[ctx.id] = ctx
                if atom is not None:
                    atom.add_contexts(contexts)

        return all_contexts
