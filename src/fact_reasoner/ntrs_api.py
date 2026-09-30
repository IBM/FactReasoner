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

# NTRS (NASA Technical Reports Server) Search API

import logging

import requests

logger = logging.getLogger(__name__)

NTRS_SEARCH_URL = "https://ntrs.nasa.gov/api/citations/search"
NTRS_BASE_URL = "https://ntrs.nasa.gov"

DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100


class NTRSAPI:
    def __init__(self):
        """
        Initialize the NTRS (NASA Technical Reports Server) API client.

        The NASA STI Repository OpenAPI is public and does not require an
        API key (see https://ntrs.nasa.gov/api/openapi/).
        """
        self.url = NTRS_SEARCH_URL

    def get_snippets(self, claim_lst, top_k: int = 1):
        """
        Retrieve search snippets for a list of claims.

        Args:
            claim_lst : list
                A list of claims (strings) for which to retrieve search snippets.
            top_k : int
                Used to size the number of results requested per query (see
                `get_search_res`). Does not truncate the returned list.
        Returns:
            dict
                A dictionary where keys are claims and values are lists of search results,
                each containing a title, snippet, and link. Records without an abstract
                are skipped since they do not provide usable evidence.
        """

        page_size = min(max(top_k, DEFAULT_PAGE_SIZE), MAX_PAGE_SIZE)

        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query, page_size=page_size)
            records = search_result.get("results", [])

            search_res_lst = []
            for record in records:
                abstract = record.get("abstract")
                if not abstract:
                    # Skip records without an abstract: they do not provide
                    # usable evidence for the FactReasoner pipeline.
                    continue

                search_res_lst.append({
                    "title": record.get("title", ""),
                    "snippet": abstract,
                    "link": self._get_link(record),
                })
            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def get_search_res(self, query, page_size: int = DEFAULT_PAGE_SIZE):
        """
        Retrieve search results for a given query from the NTRS citations search endpoint.

        Args:
            query : str
                The search query string.
            page_size : int
                Number of results to request (up to 100).
        Returns:
            dict
                The search results in JSON format.
        """

        payload = {"q": query, "page": {"size": page_size}}
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        return response.json()

    def _get_link(self, record):
        """
        Return the best available link for a citation record: the PDF
        download link if available, otherwise the citation page URL.

        Args:
            record : dict
                A single citation record from the NTRS search response.
        Returns:
            str
                The PDF link if available, otherwise the citation page URL.
        """

        downloads = record.get("downloads") or []
        if downloads:
            pdf_link = downloads[0].get("links", {}).get("pdf")
            if pdf_link:
                return NTRS_BASE_URL + pdf_link

        record_id = record.get("id", "")
        return f"{NTRS_BASE_URL}/citations/{record_id}"


if __name__ == "__main__":

    text = "Mars rover Perseverance landing site geology"

    ntrs_search = NTRSAPI()

    claim_lst = [text]
    claim_snippets = ntrs_search.get_snippets(claim_lst, top_k=3)
    print(claim_snippets)
    print("Done.")
