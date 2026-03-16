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

import logging
import re
import chromadb
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from chromadb.config import Settings as ChromaSettings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from io import BytesIO
from PyPDF2 import PdfReader
from itertools import islice
import wikipedia

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Local imports
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.search_api import SearchAPI

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

DEFAULT_COLLECTION_NAME = "lit_agent_demo"
DEFAULT_DB_PATH = "/tmp/nasa_contrib/accelerated-discovery/chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

CHARACTER_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    # keep_separator=False,
    chunk_size=1000,
    chunk_overlap=0,
)

# Regex patterns to remove common inline citation forms
CITATION_PATTERNS = [
    r"\[\d+\]",  # [1], [23]
    r"\(\d+\)",  # (1), (23)
    r"\[\s*citation\s+needed\s*\]",  # [citation needed]
    r"\(\s*citation\s+needed\s*\)",  # (citation needed)
    r"\[\s*[A-Za-z]+(?:\s+[A-Za-z]+)*\s+\d{4}\s*\]",  # [Smith 2020], [Doe et al. 2019]
    r"\^\d+",  # ^1 (superscript-like)
]


def _clean_text(text: str) -> str:
    """Apply citation removal and whitespace normalization (single line)."""
    for pattern in CITATION_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    # Collapse all whitespace (including newlines/tabs) to a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_html_paragraphs(soup: BeautifulSoup) -> str:
    """
    Extract only paragraph text (<p> tags) from HTML, removing common citation elements
    and returning a single-line string.
    """
    # Remove non-content tags globally (helps reduce clutter if nested inside <p>)
    for tag_name in ("script", "style", "noscript"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove typical citation nodes seen on many sites (e.g., Wikipedia)
    for el in soup.find_all(["sup", "span"], class_=lambda c: c and "reference" in c):
        el.decompose()
    for el in soup.select(".citation, .footnote, .ref, .references"):
        el.decompose()

    # Collect visible text only from <p> tags
    paragraphs = []
    for p in soup.find_all("p"):
        # Defensive: remove nested citation-like elements inside the paragraph
        for el in p.find_all(["sup", "span"], class_=lambda c: c and "reference" in c):
            el.decompose()
        for el in p.select(".citation, .footnote, .ref, .references"):
            el.decompose()

        para_text = p.get_text(separator=" ", strip=True)
        if para_text:
            paragraphs.append(para_text)

    # Join paragraphs into one line (no newlines), removing citations and normalizing whitespace
    text = " ".join(_clean_text(p) for p in paragraphs)
    return _clean_text(text)


def _extract_pdf_paragraphs(pdf_bytes: bytes, max_pages: int = 1) -> str:
    """
    Extract text from a PDF and return a single-line string.

    - If max_pages is provided, only the first max_pages pages are parsed.
    - Uses a simple heuristic to detect paragraphs (splitting on blank lines or multiple newlines).
    """
    reader = PdfReader(BytesIO(pdf_bytes))

    # Decide how many pages to read
    total_pages = len(reader.pages)
    pages_iter = (reader.pages[i] for i in range(total_pages))
    if max_pages is not None and max_pages > 0:
        pages_iter = islice(pages_iter, max_pages)

    pages_text = []
    for page in pages_iter:
        page_text = page.extract_text() or ""
        pages_text.append(page_text)

    full_text = "\n".join(pages_text)

    # Heuristic paragraph splitting: blank lines or >=2 newlines often separate paragraphs.
    raw_paragraphs = re.split(r"\n\s*\n|\r\n\s*\r\n|(?:\n{2,})", full_text)

    # Fallback: if no obvious paragraph boundaries, just use the whole text
    if not raw_paragraphs or len(raw_paragraphs) == 1:
        cleaned = _clean_text(full_text)
        return cleaned

    cleaned_paragraphs = [_clean_text(p) for p in raw_paragraphs if p and p.strip()]
    return _clean_text(" ".join(cleaned_paragraphs))


def extract_text_from_url(url: str, timeout: int = 10, max_pages: int = 1) -> str:
    """
    Extract text from a given URL.

    - HTML: returns ONLY the text inside <p> tags; removes citations; single-line output.
    - PDF: uses a simple paragraph heuristic; single-line output.
    - Avoids resource leaks by closing the response.

    Args:
        url (str): The web link to extract text from.
        timeout (int): Request timeout in seconds (default: 20).

    Returns:
        str: Single-line cleaned text (paragraphs-only for HTML).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    }

    response = None
    try:
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()
        content_type = (response.headers.get("Content-Type") or "").lower()

        # PDF branch
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            # Use response.content to avoid manual streaming complexity; small PDFs fit fine in memory.
            return _extract_pdf_paragraphs(response.content, max_pages=max_pages)

        # HTML branch
        soup = BeautifulSoup(response.content, "html.parser")
        return _extract_html_paragraphs(soup)

    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

    finally:
        # Ensure response is closed to avoid memory/resource leaks
        try:
            if response is not None:
                response.close()
        except Exception:
            pass


def fetch_text_from_link(link: str, max_size: int = None) -> str:
    logger.info(f"Fetching text from link: {link}")
    url_text = extract_text_from_url(url=link)
    if max_size is not None and len(url_text) > max_size:
        url_text = url_text[:max_size]
    return url_text


def get_title(text: str) -> str:
    """
    Get the title of the retrived document. By definition, the first line in the
    document is the title (we embedded them like that).
    """
    return text[: text.find("\n")]


def make_uniform(text: str) -> str:
    """
    Return a uniform representation of the text using the langchain textsplitter
    and tokensplitter tools.
    """

    character_split_texts = CHARACTER_SPLITTER.split_text(text)
    return " ".join(character_split_texts)


class ChromaReader:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
    ):
        """
        Initialize the ChromaDB.

        Args:
            collection_name: str
                The collection name in the vector database.
            persist_directory: str
                The directory used for persisting the vector database.
            embedding_model: str
                The embedding model.
            collection_metadata: dict
                A dict containing the collection metadata.
        """

        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=ChromaSettings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_collection(
            name=collection_name,
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL
            ),
        )

        print(f"[ChromaDB] initialized with {self.collection.count()} items.")

    def is_empty(self):
        return self.collection.count() == 0

    def query(self, query_texts: str, n_results: int = 5):
        """
        Returns the closests vector to the question vector

        Args:
            query_texts: str
                The user query text.
            n_results: int
                The number of results to generate.

        Returns
            The closest result to the given question.
        """
        return self.collection.query(query_texts=query_texts, n_results=n_results)


def is_content_valid(link: str, page_text: str) -> bool:
    """
    Checks if the scraped page text is likely to be valid content
    and not a restriction notice, boilerplate, or garbled/binary data.
    """
    # Initial check for empty or invalid input
    if not page_text or not isinstance(page_text, str):
        return False

    # --- Check 1: Look for specific "red flag" phrases ---
    restriction_phrases = [
        "please contact our support team",
        "cookies are used by this site",
        "copyright ©",
        "all rights are reserved",
        "ai training, and similar technologies",
        "enable javascript",
        "must have javascript enabled",
        "to continue, please verify you are a human",
        "access denied",
        "403 forbidden",
        "errors.edgesuite",
        "you do not have access to",
    ]

    text_lower = page_text.lower()
    for phrase in restriction_phrases:
        if phrase in text_lower:
            logger.warning(
                f"Redundant content detected for {link} due to restriction phrase: '{phrase}'"
            )
            return False

    if len(page_text) > 50:
        # TODO: add more filters
        replacement_char_count = page_text.count("�")
        try:
            ratio = replacement_char_count / len(page_text)
        except ZeroDivisionError:
            ratio = 0
        if ratio > 0.10:
            logger.warning(
                f"Redundant content detected for {link} due to high ratio of replacement characters: {ratio:.2%}"
            )
            return False

    # we consider the content as valid
    return True


class Retriever:
    """
    The Retriever component. We implement several versions of this component
    using a remote chromadb store (API exists), a local chromadb store, langchain
    based wikipedia retriever, and possibly others.
    """

    def __init__(
        self,
        service_type: str = "chromadb",
        collection_name: str = "wikipedia_en",
        persist_dir: str = "/tmp/wiki_db",
        top_k: int = 1,
        cache_dir: Optional[str] = None,
        fetch_text: bool = False,
        use_in_memory_vectorstore: bool = False,
        query_builder: QueryBuilder = None,
        num_workers: int = 4,
    ):
        """
        Initialize the context retriever component.

        Args:
            service_type: str
                The type of the context retriever (chromadb, wikipedia, google)
            collection_name: str
                Name of the collection of documents stored in the vectorstore
            persist_directory: str
                The dir name where the vectorstore is persisted
            top_k: int
                The top k most relevant contexts.
            cache_dir: str
                Path to the folder containing the cache (db, json).
            fetch_text: bool
                Flag to retrieve content from a link.
            use_in_memory_vectorstore: bool
                Flag to use an in memory vectorstore over chunks of retrieved texts when using Google retriever.
                Use in cases where the search results contain long documents so that they can be broken up
                into smaller chunks, which will be retrieved using the query text. When disbaled (default)
                the input will be truncated to a `max_size`.
            query_builder: QueryBuilder
                An instance of QueryBuilder to generate search queries.
            num_workers: int
                Number of threads for parallel link fetching (default: 4).
        """

        self.top_k = top_k
        self.num_workers = num_workers
        self.service_type = service_type
        self.cache_dir = cache_dir
        self.persist_dir = persist_dir
        self.fetch_text = fetch_text
        self.use_in_memory_vectorstore = use_in_memory_vectorstore
        self.query_builder = query_builder
        self.collection_name = collection_name

        self.chromadb_retriever = None
        self.langchain_retriever = None
        self.google_retriever = None
        self.in_memory_vectorstore = None

        assert self.service_type in ["chromadb", "wikipedia", "google"]

        if self.service_type == "chromadb":
            self.chromadb_retriever = ChromaReader(
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
            )
        elif self.service_type == "wikipedia":
            # Create the Wikipedia retriever. Note that page content is capped
            # at 4000 chars. The metadata has a `title` and a `summary` of the page.
            wikipedia.set_user_agent(
                "NASA IMPACT Accelerated Knowledge Discovery (AKD)/1.0 (leo@developmentseed.org)"
            )
            self.langchain_retriever = WikipediaRetriever(
                lang="en", top_k_results=top_k, wiki_client=wikipedia
            )
        elif self.service_type == "google":
            self.google_retriever = SearchAPI(cache_dir=self.cache_dir)
            if self.use_in_memory_vectorstore:
                self.in_memory_vectorstore = InMemoryVectorStore(
                    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                )
        else:
            raise ValueError(f"Unknown retriever service: {self.service_type}")

    def set_query_builder(self, query_builder: QueryBuilder = None):
        self.query_builder = query_builder

    def query(self, text: str, max_size: int = 4000) -> List[Dict[str, Any]]:
        """
        Retrieve a number of contexts relevant to the input text.

        Args:
            text: str
                The input query text.

        Returns:
            List[Dict[str, Any]]:
                The list of retrieved contexts for the input reference. A context
                is a dict with 4 keys: title, text, snippet and link.
        """

        results = []
        if self.service_type == "chromadb":
            logger.info(
                f"Retrieving {self.top_k} relevant documents for query: {text} (service: {self.service_type})"
            )

            relevant_chunks = self.chromadb_retriever.query(
                query_texts=[text],
                n_results=self.top_k,
            )

            docs = relevant_chunks.get("documents", [[]])[0]
            metadatas = relevant_chunks.get("metadatas", [[]])[0]

            passages = []
            for doc_content, metadata in zip(docs, metadatas):
                cleaned = _clean_text(doc_content)
                first_sentence, _, _ = cleaned.partition(". ")
                snippet = (first_sentence + ".") if first_sentence and _ else cleaned[:200]
                passage = {
                    "title": metadata.get("title", "No Title Provided"),
                    "text": make_uniform(cleaned),
                    "snippet": snippet,
                    "link": metadata.get("source", ""),
                }
                passages.append(passage)

            results.extend(passages)
        elif self.service_type == "wikipedia":
            logger.info(
                f"Retrieving {self.top_k} relevant documents for query: {text} (service: {self.service_type})"
            )

            passages = []

            # Get most relevant docs to the query
            try:
                rel_docs = self.langchain_retriever.invoke(text)
            except Exception as e:
                logger.warning(f"Wikipedia retrieval failed for query '{text}': {e}")
                return results
            for doc in rel_docs:
                title = doc.metadata["title"]
                summary = doc.metadata["summary"]
                link = doc.metadata["source"]
                doc_content = make_uniform(doc.page_content)
                passages.append(
                    dict(title=title, text=doc_content, snippet=summary, link=link)
                )

            # Extract the top_k passages
            n = min(self.top_k, len(passages))
            for i in range(n):
                results.append(
                    passages[i]
                )  # a passage is a dict with title and text as keys

        elif self.service_type == "google":
            logger.info(
                f"Retrieving {self.top_k} search results for: {text} (service: {self.service_type})"
            )

            if not text:
                return results  # empty list

            # Generate the query text if there is a query builder
            if self.query_builder is not None:
                query_text = self.query_builder.run(text)
            else:
                query_text = text

            # Truncate the text if too long (for Google)
            query_text = query_text if len(query_text) < 2048 else query_text[:2048]
            logger.info(f"Using query text: {query_text}")
            passages = []

            # Get the search results
            search_results = self.google_retriever.get_snippets([query_text])

            n = len(search_results[query_text])

            # If no hits then relax query by removing specific '"' (if any)
            if n == 0:  # no hits
                query_text = query_text.replace('"', "")  # relax query text
                search_results = self.google_retriever.get_snippets([query_text])

            search_hits = search_results[query_text]
            n = len(search_hits)

            # --- Parallel fetch all links ---
            if self.fetch_text:
                max_sz = None if self.use_in_memory_vectorstore else max_size
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    fetch_futures = {
                        executor.submit(fetch_text_from_link, hit["link"], max_sz): i
                        for i, hit in enumerate(search_hits)
                    }
                    fetched_texts = [""] * n
                    for future in as_completed(fetch_futures):
                        idx = fetch_futures[future]
                        try:
                            fetched_texts[idx] = future.result()
                        except Exception:
                            fetched_texts[idx] = ""
            else:
                fetched_texts = [""] * n

            # --- Select top_k valid results ---
            passages = []
            index_available = []
            count_content = 0

            for i, hit in enumerate(search_hits):
                if count_content >= self.top_k:
                    break

                title, snippet, link = hit["title"], hit["snippet"], hit["link"]
                raw_page_text = fetched_texts[i]

                if self.fetch_text:
                    if not is_content_valid(link, raw_page_text):
                        raw_page_text = ""
                    doc_content = make_uniform(raw_page_text) if raw_page_text else ""

                    if not doc_content or any(
                        kw in doc_content.lower()
                        for kw in ("chatgpt", "factscore", "dataset viewer")
                    ):
                        doc_content = ""
                        index_available.append(i)
                        continue

                    if self.use_in_memory_vectorstore:
                        split_doc_content = CHARACTER_SPLITTER.split_text(doc_content)
                        documents = [
                            Document(
                                id=f"{doc_id}",
                                page_content=chunk,
                                metadata={"source": link},
                            )
                            for doc_id, chunk in enumerate(split_doc_content)
                        ]
                        self.in_memory_vectorstore.add_documents(documents=documents)
                        retriever = self.in_memory_vectorstore.as_retriever(
                            search_kwargs={"k": 3}
                        )
                        retrieved_docs = retriever.invoke(query_text)
                        doc_content = "\n\n".join(
                            [doc.page_content for doc in retrieved_docs]
                        )

                    count_content += 1
                else:
                    doc_content = ""
                    count_content += 1

                passages.append(
                    dict(title=title, text=doc_content, snippet=snippet, link=link)
                )

            # --- Fallback to empty-content entries ---
            if count_content < self.top_k:
                for i in index_available:
                    if count_content >= self.top_k:
                        break
                    hit = search_hits[i]
                    passages.append(
                        dict(
                            title=hit["title"],
                            text="",
                            snippet=hit["snippet"],
                            link=hit["link"],
                        )
                    )
                    count_content += 1

            results.extend(passages)
            logger.info(f"Retrieved {len(results)} results for query.")
        return results


class ContextRetriever:
    """
    Parallel context retriever that wraps a ContextRetriever and dispatches
    retrieval tasks across a thread pool.
    """

    def __init__(
        self,
        retriever: Retriever,
        context_summarizer: Optional[ContextSummarizer] = None,
        num_workers: int = 4,
    ):
        self.retriever = retriever
        self.context_summarizer = context_summarizer
        self.num_workers = num_workers

    def _retrieve_for_item(
        self,
        text: str,
        atom: Optional[Atom] = None,
        id_prefix: str = "c",
    ) -> List[Context]:
        """Worker function: retrieve contexts (and optionally summarize) for one item."""
        retrieved = self.retriever.query(text=text)

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
        if (
            self.context_summarizer is not None
            and atom is not None
            and len(contexts) > 0
        ):
            results = asyncio.run(
                self.context_summarizer.run_batch(
                    [c.get_text() for c in contexts], atom.text
                )
            )
            for context, result in zip(contexts, results):
                if result["summary"]:
                    context.set_synthetic_summary(result["summary"])
                    context.set_probability(
                        result["probability"] * context.get_probability()
                    )
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
                future = executor.submit(self._retrieve_for_item, query, None, "c_q")
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
