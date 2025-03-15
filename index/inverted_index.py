import os
import time
import pickle
from collections import defaultdict
from typing import List, Dict, Set
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from data.document import Document
from data.preprocessing import tokenize, remove_stopwords, stem_tokens, preprocess
from .trie import Trie
from .bitmap_index import BitmapIndex

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Cache files and versioning
CACHE_INDEX_FILE = os.path.join(os.path.dirname(__file__), "inverted_index_cache.pkl")
INDEX_VERSION = "v2"  # Update version when making structural changes to the index


def _build_partial_index(documents: List[Document]) -> Dict[str, Dict[str, List[int]]]:
    """
    Worker function to build a partial inverted index from a list of documents.
    
    This function is executed in a separate process for parallel indexing.
    
    Args:
        documents: List of documents to index
        
    Returns:
        Partial inverted index mapping terms to document IDs and positions
    """
    partial_index = defaultdict(dict)
    for document in documents:
        combined_text = f"{document.title} {document.content}"
        tokens = tokenize(combined_text)
        tokens = remove_stopwords(tokens)
        tokens = stem_tokens(tokens)
        
        for pos, term in enumerate(tokens):
            if document.doc_id not in partial_index[term]:
                partial_index[term][document.doc_id] = []
            partial_index[term][document.doc_id].append(pos)
            
    return partial_index


class InvertedIndex:
    """
    Inverted index data structure for efficient text search.
    
    Includes positional information, trie for prefix matching, and bitmap index
    for ultra-fast boolean operations.
    
    Attributes:
        index: The core inverted index mapping terms to documents and positions
        version: Version string for cache compatibility
        trie: Trie data structure for prefix and fuzzy matching
        bitmap_index: Bitmap representation for fast boolean operations
        doc_ids: List of all document IDs in the index
    """
    def __init__(self):
        self.index: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.version = INDEX_VERSION
        self.trie = Trie()
        self.bitmap_index = None
        self.doc_ids = []
        
    def index_document(self, document: Document) -> None:
        """
        Index a single document (sequential version).
        
        Updates the inverted index and trie index.
        
        Args:
            document: Document to index
        """
        combined_text = f"{document.title} {document.content}"
        tokens = tokenize(combined_text)
        tokens = remove_stopwords(tokens)
        tokens = stem_tokens(tokens)
        
        # Track unique document IDs for bitmap index
        if document.doc_id not in self.doc_ids:
            self.doc_ids.append(document.doc_id)
            
        for pos, term in enumerate(tokens):
            # Update inverted index
            if document.doc_id not in self.index[term]:
                self.index[term][document.doc_id] = []
            self.index[term][document.doc_id].append(pos)
            
            # Update trie for fast prefix matches
            self.trie.insert(term, document.doc_id, pos)
    
    def build_index(self, documents: List[Document], parallel: bool = True, workers: int = 4) -> None:
        """
        Build the inverted index from a list of documents.
        
        Args:
            documents: List of documents to index
            parallel: Whether to use parallel processing
            workers: Number of worker processes to use
        """
        start_time = time.time()
        
        if not parallel:
            # Sequential indexing
            for doc in documents:
                self.index_document(doc)
            logger.info(f"Inverted index built with {len(self.index)} terms (sequential).")
        else:
            # Parallel indexing
            chunk_size = max(1, len(documents) // workers)
            chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]
            partial_indices = []
            
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(_build_partial_index, chunk) for chunk in chunks]
                for future in as_completed(futures):
                    partial_indices.append(future.result())
                    
            # Merge partial index dictionaries
            for part in partial_indices:
                for term, postings in part.items():
                    if term in self.index:
                        for doc_id, positions in postings.items():
                            if doc_id in self.index[term]:
                                self.index[term][doc_id].extend(positions)
                            else:
                                self.index[term][doc_id] = positions
                    else:
                        self.index[term] = postings
                        
            logger.info(f"Merged partial indices from {len(partial_indices)} workers. Total terms: {len(self.index)}.")
            
        # Build bitmap index for ultra-fast boolean operations
        self._build_bitmap_index()
        
        logger.info(f"Completed full indexing in {time.time() - start_time:.2f} seconds")
    
    def _build_bitmap_index(self) -> None:
        """
        Build bitmap representation of the index for faster boolean operations.
        
        This is how commercial search engines achieve sub-millisecond conjunctive queries.
        """
        start_time = time.time()
        
        if not self.doc_ids:
            return
            
        logger.info("Building bitmap index for fast boolean operations...")
        self.bitmap_index = BitmapIndex(len(self.doc_ids))
        
        # Register document IDs
        for idx, doc_id in enumerate(self.doc_ids):
            self.bitmap_index.add_document(doc_id, idx)
            
        # Add terms to bitmap index
        for term, postings in self.index.items():
            self.bitmap_index.add_term(term, list(postings.keys()))
            
        logger.info(f"Built bitmap index in {time.time() - start_time:.2f} seconds")

    def save_to_file(self, filepath: str = CACHE_INDEX_FILE) -> None:
        """
        Save index structures to disk for persistence between runs.
        
        Args:
            filepath: Path to save the index cache
        """
        start_time = time.time()
        
        with open(filepath, "wb") as f:
            pickle.dump({
                "version": self.version, 
                "index": self.index,
                "doc_ids": self.doc_ids
            }, f)
            
        logger.info(f"Saved inverted index cache to {filepath} in {time.time() - start_time:.2f} seconds")

    def load_from_file(self, filepath: str = CACHE_INDEX_FILE) -> bool:
        """
        Load index from disk and rebuild auxiliary structures.
        
        Args:
            filepath: Path to the index cache file
            
        Returns:
            True if successfully loaded, False otherwise
        """
        if os.path.exists(filepath):
            start_time = time.time()
            
            with open(filepath, "rb") as f:
                cached = pickle.load(f)
                
            if cached.get("version") == self.version:
                self.index = cached.get("index", defaultdict(dict))
                self.doc_ids = cached.get("doc_ids", [])
                
                # Rebuild trie and bitmap indexes
                logger.info("Rebuilding trie from loaded index...")
                for term, postings in self.index.items():
                    for doc_id, positions in postings.items():
                        for position in positions:
                            self.trie.insert(term, doc_id, position)
                
                self._build_bitmap_index()
                
                logger.info(f"Loaded index with {len(self.index)} terms in {time.time() - start_time:.2f} seconds")
                return True
            else:
                logger.info("Index cache version mismatch. Rebuilding index.")
                
        return False

    def search(self, query: str) -> Set[str]:
        """
        Perform a basic AND boolean search for the query.
        
        Returns documents that contain all query terms.
        
        Args:
            query: Search query string
            
        Returns:
            Set of document IDs matching the query
        """
        start_time = time.time()
        query_tokens = preprocess(query)
        
        if not query_tokens:
            return set()
        
        # Use bitmap index for faster boolean operations if available
        if self.bitmap_index:
            result = self.bitmap_index.boolean_and(query_tokens)
            logger.debug(f"Bitmap search completed in {time.time() - start_time:.4f} seconds")
            return result
        
        # Fall back to original intersection method if bitmap index not available
        result_set = None
        for term in query_tokens:
            postings = set(self.index.get(term, {}).keys())
            result_set = postings if result_set is None else result_set & postings
            
        logger.debug(f"Standard search completed in {time.time() - start_time:.4f} seconds")
        return result_set if result_set else set()
    
    def get_term_highlights(self, doc_id: str, query: str) -> List[int]:
        """
        Get positions where query terms appear in a document for highlighting.
        
        Used for generating search snippets with highlighted terms.
        
        Args:
            doc_id: Document ID to get positions for
            query: Search query string
            
        Returns:
            List of positions where query terms appear
        """
        query_tokens = preprocess(query)
        positions = []
        
        for term in query_tokens:
            if term in self.index and doc_id in self.index[term]:
                positions.extend(self.index[term][doc_id])
                
        return sorted(positions)
    
    def get_candidate_docs(self, query: str) -> Set[str]:
        """
        Get candidate documents for a query using bitmap operations.
        
        This is much faster than scoring all documents and provides a candidate set
        for ranking algorithms.
        
        Args:
            query: Search query string
            
        Returns:
            Set of candidate document IDs
        """
        query_tokens = preprocess(query)
        if not query_tokens:
            return set()
            
        # Use bitmap index for ultra-fast retrieval of potential matches
        if self.bitmap_index:
            # First try AND (all terms must be present)
            candidates = self.bitmap_index.boolean_and(query_tokens)
            
            # If too few results, try OR (any term can be present)
            if len(candidates) < 10:
                candidates = self.bitmap_index.boolean_or(query_tokens)
            
            return candidates
            
        # Fall back to standard retrieval if bitmap index not available
        # Find documents with any query term (OR)
        candidates = set()
        for term in query_tokens:
            if term in self.index:
                candidates.update(self.index[term].keys())
                
        return candidates
    
    def find_similar_words(self, word: str, max_distance: int = 1) -> List[str]:
        """
        Find words similar to the given word using the trie.
        
        Uses prefix filtering and Levenshtein distance for fuzzy matching.
        
        Args:
            word: Word to find similar words for
            max_distance: Maximum edit distance between words
            
        Returns:
            List of similar words
        """
        # First check exact matches and prefixes
        similar_words = []
        if hasattr(self, 'trie'):
            # Exact match
            if self.trie.search(word):
                similar_words.append(word)
                
            # Words with the same prefix (first 3 chars)
            if len(word) >= 3:
                prefix = word[:3]
                prefix_words = self.trie.starts_with(prefix)
                
                # Filter by edit distance
                for pw in prefix_words:
                    if pw != word and levenshtein_distance(word, pw) <= max_distance:
                        similar_words.append(pw)
        
        # Fallback to slower method if trie isn't helping or available
        if not similar_words:
            # Try the top 1000 most frequent terms by document frequency
            top_terms = sorted(self.index.keys(), 
                              key=lambda t: len(self.index[t]), 
                              reverse=True)[:1000]
            similar_words = [t for t in top_terms 
                           if levenshtein_distance(word, t) <= max_distance]
        
        return similar_words


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein (edit) distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance (number of insertions, deletions, or substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
        
    if len(s2) == 0:
        return len(s1)
        
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]


if __name__ == "__main__":
    from data.loader import load_arxiv_dataset
    docs = load_arxiv_dataset("path/to/arxiv_dataset.json")
    index = InvertedIndex()
    if not index.load_from_file():
        index.build_index(docs, parallel=True, workers=4)
        index.save_to_file()
    query = input("Enter search query: ")
    results = index.search(query)
    print("Results:", results)