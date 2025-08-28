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

import io
import re
from typing import Callable, List, Optional

import chromadb
import html2text
import requests
import torch

# from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from bs4 import BeautifulSoup
from bs4.element import Tag
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

from src.fact_reasoner.query_builder import QueryBuilder
from src.fact_reasoner.search_api import SearchAPI

COLLECTION_NAME = "wikipedia_en"
DB_PATH = "/home/radu/wiki_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEWLINES_RE = re.compile(r"\n{2,}")  # two or more "\n" characters

CHARACTER_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    #keep_separator=False,
    chunk_size=1000,
    chunk_overlap=0
)

# TOKEN_SPLITTER = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)

def remove_citation(paragraph: str) -> str:
    """Remove all citations (numbers in side square brackets) in paragraph"""
    return re.sub(r'\[\d+\]', '', paragraph)

def remove_new_line(paragraph: str) -> str:
    return paragraph.replace("\n", "")

def compose_fns(functions: List[Callable]) -> Callable:
    def ret(input):
        for fn in functions:
            input = fn(input)

        return input

    return ret

def html_to_text2(html_text: str) -> str:
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True
    text_maker.ignore_tables = True
    text_maker.ignore_images = True
    text_maker.emphasis_mark = ''
    text_maker.body_width = 0
    text = text_maker.handle(html_text).replace('\n', '').strip()
    return text

def html_to_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, 'html.parser')

    preprocess_fn = compose_fns([remove_citation, remove_new_line])
    paragraphs = list(map(lambda p: preprocess_fn(p.getText()), soup.find_all('p')))
    return "\n".join(paragraphs)

def fetch_text_from_link(link: str, max_size: int = None) -> str:
    print(f"Fetching text from link: {link}")
    try:
        if link.endswith('.pdf'): # pdf page
            r = requests.get(link, timeout=10)
            f = io.BytesIO(r.content)

            reader = PdfReader(f)
            pdf_texts = [p.extract_text().strip() for p in reader.pages]

            # Filter the empty strings
            pdf_texts = [text for text in pdf_texts if text]
            contents = "\n".join(pdf_texts)


        else: # html page
            html = requests.get(link, timeout=10).text
            contents = html_to_text(html)
    
        if max_size is not None and len(contents) > max_size:
            contents = contents[:max_size]
    except Exception as _:
        contents = ""
    return contents

def get_title(text: str) -> str:
    """
    Get the title of the retrived document. By definition, the first line in the
    document is the title (we embedded them like that).
    """
    return text[:text.find("\n")]

def make_uniform(text: str) -> str:
    """
    Return a uniform representation of the text using the langchain textsplitter
    and tokensplitter tools.
    """

    character_split_texts = CHARACTER_SPLITTER.split_text(text)
    # torch.cuda.empty_cache()
    return " ".join(character_split_texts)

    # token_split_texts = []
    # for text in character_split_texts:
    #     token_split_texts += TOKEN_SPLITTER.split_text(text)
    # torch.cuda.empty_cache()

    # return " ".join(token_split_texts)

class ChromaReader:
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str,
        collection_metadata: dict = None,
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
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model,
            device=self.device,
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata=collection_metadata,
        )

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
  

class ContextRetriever:
    """
    The ContextRetriever component. We implement several versions of this component
    using a remote chromadb store (API exists), a local chromadb store, langchain
    based wikipedia retriever, and possibly others.
    """
    
    def __init__(
            self,
            db_address: str = "9.59.197.15",
            db_port: int = "5000",
            db_remote: bool = False,
            service_type: str = "chromadb",
            top_k: int = 1,
            cache_dir: Optional[str] = None,
            debug: bool = False,
            fetch_text: bool = False,
            use_in_memory_vectorstore: bool = False,
            query_builder: QueryBuilder = None
    ):
        """
        Initialize the context retriever component.

        Args:
            db_address: str
                The IP address of the host running the API to the vector store.
            db_port: int
                The port of the host running the API for querying the vector store.
            db_remote: bool
                Flag indicating a remote (http) service.
            service_type: str
                The type of the context retriever (chromadb, langchain, google)
            top_k: int
                The top k most relevant contexts.
            cache_dir: str
                Path to the folder containing the cache (db, json).
            debug: bool
                Flag to set debugging mode.
            fetch_text: bool
                Flag to retrieve content from a link.
            use_in_memory_vectorstore: bool
                Flag to use an in memory vectorstore over chunks of retrieved texts when using Google retriever.
                Use in cases where the search results contain long documents so that they can be broken up
                into smaller chunks, which will be retrieved using the query text. When disbaled (default)
                the input will be truncated to a `max_size`.
            query_builder: QueryBuilder
                An instance of QueryBuilder to generate search queries.
        """
        
        self.top_k = top_k
        self.db_address = db_address
        self.db_port = db_port
        self.db_remote = db_remote
        self.service_type = service_type
        self.chromadb_retriever = None
        self.langchain_retriever = None
        self.google_retriever = None
        self.in_memory_vectorstore = None
        self.debug = debug
        self.cache_dir = cache_dir
        self.fetch_text = fetch_text
        self.use_in_memory_vectorstore = use_in_memory_vectorstore
        self.query_builder = query_builder

        assert self.service_type in ["chromadb", "langchain", "google"]

        if self.service_type == "chromadb":
            if not self.db_remote: # Create a local ChromaDB client
                self.chromadb_retriever = ChromaReader(
                    collection_name=COLLECTION_NAME, 
                    persist_directory=DB_PATH, 
                    embedding_model=EMBEDDING_MODEL, 
                    collection_metadata={"hnsw:space": "cosine"}
                )
        elif self.service_type == "langchain":
            # Create the Wikipedia retriever. Note that page content is capped
            # at 4000 chars. The metadata has a `title` and a `summary` of the page.
            self.langchain_retriever = WikipediaRetriever(lang="en", top_k_results=top_k)
        elif self.service_type == "google":
            self.google_retriever = SearchAPI(cache_dir=self.cache_dir)
            if self.use_in_memory_vectorstore:
                self.in_memory_vectorstore = InMemoryVectorStore(
                    HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
                )

    def set_query_builder(self, query_builder: QueryBuilder = None):
        self.query_builder = query_builder
    
    def query(
            self, 
            text: str,
    ) -> List[str]:
        """
        Retrieve a number of contexts relevant to the input text.

        Args:
            text: str
                The input query text.

        Returns:
            List[dict]
                The list of retrieved contexts for the input reference. A context
                is a dict with 4 keys: title, text, snippet and link.
        """

        results = []
        if self.service_type == "chromadb":
            if self.db_remote:
                if self.debug:
                    print(f"Retrieving {self.top_k} relevant documents for query: {text}")
                    print(f"Using chromadb")

                # Send a POST request with JSON data using the session object
                headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
                url = "http://" + self.host_address + ":" + str(self.host_port) + "/query"
                data = dict(
                    query_text=text, 
                    n_results=self.top_k,
                )

                # Get the response
                with requests.Session() as s:
                    response = s.post(url, json=data, headers=headers)
                    results = []
                    if response.status_code == 200: # success
                        passages = response.json()
                        results = [passages[str(i)] for i in range(len(passages))]
            else: # local
                if self.debug:
                    print(f"Retrieving {self.top_k} relevant documents for query: {text}")
                    print(f"Using chromadb")

                # Retrieve the relevant chunks from the vector store
                relevant_chunks = self.chromadb_retriever.query(
                    query_texts=[text],
                    n_results=self.top_k,
                )

                # Get the chunks (documents)
                docs = relevant_chunks["documents"][0]
                passages = [dict(title=get_title(doc), text=make_uniform(doc), snippet="", link="") for doc in docs]

                n = min(self.top_k, len(passages))
                for i in range(n):
                    results.append(passages[i]) # a passage is a dict with title and text as keys
        elif self.service_type == "langchain":
            if self.debug:
                print(f"Retrieving {self.top_k} relevant documents for query: {text}")
                print(f"Using langchain WikipediaRetriever")

            passages = []                

            # Get most relevant docs to the query
            rel_docs = self.langchain_retriever.invoke(text)
            for doc in rel_docs:
                title = doc.metadata["title"]
                summary = doc.metadata["summary"]
                link = doc.metadata["source"]
                doc_content = make_uniform(doc.page_content)
                passages.append(dict(title=title, text=doc_content, snippet=summary, link=link))
            
            # Extract the top_k passages
            n = min(self.top_k, len(passages))
            for i in range(n):
                results.append(passages[i]) # a passage is a dict with title and text as keys
        elif self.service_type == "google":
            print(f"Retrieving {self.top_k} search results for: {text}")
            if not text:
                return results

            # Generate the query text if there is a query builder
            if self.query_builder is not None:
                result = self.query_builder.run(text)
                query_text = result.get("query", text)
            else:
                query_text = text
            
            # Truncate the text if too long (for Google)
            query_text = query_text if len(query_text) < 2048 else query_text[:2048]
            print(f"Using query text: {query_text}")
            passages = []

            # Get the search results
            search_results = self.google_retriever.get_snippets([query_text])

            # n = min(self.top_k, len(search_results[query_text]))
            n = len(search_results[query_text])

            # If no hits then relax query by removing specific '"' (if any)
            if n == 0: # no hits
                query_text = query_text.replace('"', '') # relax query text
                search_results = self.google_retriever.get_snippets([query_text])

            n = len(search_results[query_text])

            i = 0
            cont_content = 0
            index_available = []

            while ((i < n) and (cont_content < self.top_k)):
                # we retrieve content from the link
                if self.fetch_text:
                    # loop to check that the content retrieved is not empty: if it is empty, check the next link
                    while (i < n):
                        res = search_results[query_text][i]
                        title = res['title']
                        snippet = res['snippet']
                        link = res['link']

                        # if using in memory vector store, do not set a max size initially on the page text
                        # it will be determined by the splitter chunk size and number of chunks.
                        page_text = fetch_text_from_link(link, max_size=None if self.use_in_memory_vectorstore else 4000)
                        doc_content = make_uniform(page_text) if len(page_text) > 0 else ""
                        
                        if self.use_in_memory_vectorstore:
                            # make documents for vectorstore
                            split_doc_content = CHARACTER_SPLITTER.split_text(doc_content)
                            documents = [Document(id=f"{doc_id}", page_content=text, metadata={"source": link})
                                for doc_id, text in enumerate(split_doc_content)
                            ]
                            self.in_memory_vectorstore.add_documents(documents=documents)
                            retriever = self.in_memory_vectorstore.as_retriever(search_kwargs={"k": 3})
                            retrieved_docs = retriever.invoke(query_text)
                            doc_content = "\n\n".join([doc.page_content for doc in retrieved_docs])
         
                        # no content from the link
                        # or the content may be AI-generated or talking about a dataset used to test LLMs                      
                        # we store the index in case we run out of links and we have to come back to the previous ones
                        if (doc_content == "") or ("chatgpt" in doc_content.lower()) or ("factscore" in doc_content.lower()) or ("dataset viewer" in doc_content.lower()):
                            doc_content = ""
                            index_available.append(i)
                            i += 1

                        # found content from the link
                        else: 
                            cont_content += 1
                            i += 1
                            break

                # we do not retrieve content from the link
                else:
                    res = search_results[query_text][i]
                    title = res['title']
                    snippet = res['snippet']
                    link = res['link']
                    doc_content = ""
                    i += 1
                    cont_content += 1

                passages.append(dict(title=title, text=doc_content, snippet=snippet, link=link))

            # in case we run out of links and we have to come back to the previous ones, whose content is empty
            if (cont_content < self.top_k):
                for i in index_available:
                    res = search_results[query_text][i]
                    title = res['title']
                    snippet = res['snippet']
                    link = res['link']
                    doc_content = ""

                    passages.append(dict(title=title, text=doc_content, snippet=snippet, link=link))

                    cont_content += 1
                    if cont_content == self.top_k:
                        break

            for passage in passages:
                results.append(passage) # a passage is a dict with title and text as keys

        return results
        
    
def test_html_to_text():
    import sys

    #link = "https://m.imdb.com/title/tt0098844/fullcredits/cast/?ref_=tt_cl_sm"
    #link = "https://www.washingtonpost.com/history/2024/06/26/treasure-hunter-was-ready-retire-then-he-found-hundreds-coins/"
    link = "https://olympics.com/en/athletes/roxana-diaz"

    html = requests.get(link, timeout=10).text
    contents = html_to_text(html)
    print(contents)
    sys.exit(0)

def test_langchain():
    import sys

    from langchain_community.document_loaders import AsyncHtmlLoader
    from langchain_community.document_transformers import Html2TextTransformer

    urls = [
        "https://www.imdb.com/name/nm0280890/",
        "https://en.wikipedia.org/wiki/Lanny_Flaherty",
        "https://m.imdb.com/title/tt0098844/fullcredits/cast/?ref_=tt_cl_sm",
        "https://www.tvguide.com/celebrities/lanny-flaherty/credits/3030406905/",
        "https://www.fandango.com/people/lanny-flaherty-220003"
    ]

    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    print(docs_transformed[0].page_content)
    print("****" * 20)

    print(docs_transformed[1].page_content)
    print("****" * 20)

    print(docs_transformed[2].page_content)
    print("****" * 20)

    print(docs_transformed[3].page_content)
    print("****" * 20)

    print(docs_transformed[4].page_content)
    print("****" * 20)
    sys.exit(0)

if __name__ == '__main__':

    query = "Lanny Flaherty has appeared in Law & Order."
    cache_dir = "my_database.db"
    query_builder = QueryBuilder(model="llama-3.1-70b-instruct")

    retriever = ContextRetriever(
        top_k=5,
        service_type="google",
        cache_dir=cache_dir,
        query_builder=query_builder
    )
    
    contexts = retriever.query(text=query)
    
    print(f"Number of contexts: {len(contexts)}")
    for context in contexts:
        print(context)
        print("****" * 20)

    print("Done.")
