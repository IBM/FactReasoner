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

# Search API

import json
import logging
import os
import sqlite3
from ast import literal_eval

import requests
from dotenv import load_dotenv
from thefuzz import fuzz

logger = logging.getLogger(__name__)

class SearchAPI():
    def __init__(self, cache_dir: str = None, similarity_threshold: float = 90):

        if not os.environ.get("_DOTENV_LOADED"):
            load_dotenv(override=True) 
            os.environ["_DOTENV_LOADED"] = "1"

        self.serper_key = os.getenv("SERPER_API_KEY")
        self.url = "https://google.serper.dev/search"
        self.headers = {'X-API-KEY': self.serper_key,
                        'Content-Type': 'application/json'}

        # Set up database
        if cache_dir is None:
            cache_dir = "./db/google_cache.db"

        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with FTS5 for full-text search, using WAL mode for better concurrency."""
        with sqlite3.connect(self.cache_dir) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")  # Enable Write-Ahead Logging
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS search_cache USING fts5(
                    query, response
                )
            """)
            conn.commit()

    def get_snippets(self, claim_lst):
        text_claim_snippets_dict = {}
        for query in claim_lst:
            search_result = self.get_search_res(query)
            if "statusCode" in search_result:
                logger.error(search_result['message'])
                exit()
            organic_res = search_result.get("organic", [])

            search_res_lst = [{"title": item.get("title", ""),
                               "snippet": item.get("snippet", ""),
                               "link": item.get("link", "")}
                              for item in organic_res]

            text_claim_snippets_dict[query] = search_res_lst
        return text_claim_snippets_dict

    def get_search_res(self, query):
        cache_key = query.strip()
        cached_response = self._get_from_cache(cache_key)
        if cached_response:
            logger.info(f"CACHE HIT! key={cache_key}, q={cached_response['searchParameters']['q']}")
            return cached_response

        # Make API request
        payload = json.dumps({"q": query})
        response = requests.request("POST", self.url, params={"num": 15}, headers=self.headers, data=payload)
        response_json = literal_eval(response.text)
        try:
            self._save_to_cache(cache_key, response_json)
        except sqlite3.Error as e:
            logger.error(f"Error saving to cache: {e}. Continuing...")
        return response_json

    def _get_from_cache(self, query):
        """Retrieve the top 3 most relevant cached search results and use fuzz.token_sort_ratio to pick the best one."""
        # Escape double quotes for FTS5
        query = query.replace('"', '""')

        with sqlite3.connect(self.cache_dir) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT query, response FROM search_cache
                WHERE query MATCH ?
                LIMIT 3
            """, (f'"{query}"',))
            rows = cursor.fetchall()

        if not rows:
            return None

        # Compute similarity scores using fuzz.token_sort_ratio
        best_match = max(rows, key=lambda row: fuzz.token_sort_ratio(query, row[0]))
        best_score = fuzz.token_sort_ratio(query, best_match[0])

        return json.loads(best_match[1]) if best_score > self.similarity_threshold else None

    def _save_to_cache(self, query, response_json):
        """Save a search result to the SQLite FTS5 cache with transactions."""
        # Do not cache empty search results
        organic_res = response_json.get("organic", [])
        if len(organic_res) == 0:
            return
        
        conn = sqlite3.connect(self.cache_dir)
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN TRANSACTION;")  # Start transaction
            cursor.execute("REPLACE INTO search_cache (query, response) VALUES (?, ?)",
                           (query, json.dumps(response_json)))
            conn.commit()  # Commit transaction
        except sqlite3.Error as e:
            conn.rollback()  # Rollback on error
            logger.error(f"Database error: {e}")
        finally:
            conn.close()


if __name__ == '__main__':
    
    cache_dir = "my_database.db"

    text = "Neil B. Todd was an American geneticist"
    # text = "Lanny Flaherty is an American."
    # text = "\"Benefits of displaying menu prices without tax included\" OR \"Advantages of separate tax display in restaurant menus\""
    # text = text.replace('"', '')

    # Initialize search api
    web_search = SearchAPI(cache_dir=cache_dir)

    claim_lst = [text]
    claim_snippets = web_search.get_snippets(claim_lst)
    print(claim_snippets)
    print("Done.")
    print("Done.")
