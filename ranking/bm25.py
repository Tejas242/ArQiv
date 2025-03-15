import math
from typing import List, Dict, Set
from data.document import Document
from data.preprocessing import tokenize, remove_stopwords, stem_tokens, preprocess
from index.inverted_index import InvertedIndex


class BM25Ranker:
    """
    BM25 ranking algorithm implementation.
    
    BM25 (Best Matching 25) is a ranking function used to rank documents by relevance.
    It extends TF-IDF with document length normalization and term saturation.
    
    Attributes:
        documents: List of documents to rank
        index: Inverted index for term lookups
        k1: Term saturation parameter
        b: Document length normalization parameter
        N: Number of documents in collection
        doc_lengths: Dictionary of document lengths
        avgdl: Average document length
        idf_cache: Cache for IDF values
    """
    def __init__(self, documents: List[Document], index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.index = index
        self.k1 = k1
        self.b = b
        self.N = len(documents)
        self.doc_lengths: Dict[str, int] = {}
        self.avgdl: float = 0.0
        self.idf_cache: Dict[str, float] = {}  # Cache for IDF values
        self._compute_doc_lengths()

    def _compute_doc_lengths(self) -> None:
        """
        Compute document lengths and average document length.
        
        These are used for length normalization in the BM25 formula.
        """
        total_length = 0
        for doc in self.documents:
            combined_text = doc.title + " " + doc.content
            tokens = tokenize(combined_text)
            tokens = remove_stopwords(tokens)
            tokens = stem_tokens(tokens)
            length = len(tokens)
            self.doc_lengths[doc.doc_id] = length
            total_length += length
        self.avgdl = total_length / self.N if self.N else 0

    def _idf(self, term: str) -> float:
        """
        Calculate Inverse Document Frequency (IDF) with caching.
        
        Args:
            term: Term to calculate IDF for
            
        Returns:
            IDF value for the term
        """
        if term in self.idf_cache:
            return self.idf_cache[term]
            
        df = len(self.index.index.get(term, {}))
        idf_val = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        self.idf_cache[term] = idf_val
        return idf_val

    def precompute_idf(self, query: str) -> None:
        """
        Precompute IDF values for all query terms.
        
        Args:
            query: Query string to precompute IDF values for
        """
        query_tokens = preprocess(query)
        for term in query_tokens:
            if term not in self.idf_cache:
                self._idf(term)
                
    def score_document(self, doc_id: str, query: str) -> float:
        """
        Compute BM25 score for a single document.
        
        Used for individual document scoring, often in parallel processing.
        
        Args:
            doc_id: Document ID to score
            query: Raw query string
            
        Returns:
            BM25 score for the document
        """
        query_tokens = preprocess(query)
        if not query_tokens:
            return 0.0
            
        score = 0.0
        dl = self.doc_lengths.get(doc_id, 0)
        if dl == 0:
            return 0.0
            
        for term in query_tokens:
            if term in self.index.index and doc_id in self.index.index[term]:
                f = len(self.index.index[term][doc_id])  # term frequency
                idf = self._idf(term)
                term_score = idf * ((f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))))
                score += term_score
                
        return score

    def rank(self, query: str) -> Dict[str, float]:
        """
        Rank documents for the given query using BM25.
        
        This uses candidate prefiltering via bitmap index for efficiency.
        
        Args:
            query: Query string to rank documents against
            
        Returns:
            Dictionary mapping document IDs to BM25 scores
        """
        # Precompute IDF for all query terms at once for efficiency
        self.precompute_idf(query)
        
        query_tokens = preprocess(query)
        scores: Dict[str, float] = {}
        
        # Get candidate documents that may match (via bitmap index)
        candidates = self.index.get_candidate_docs(query)
        
        # Score only candidate documents for improved performance
        for doc_id in candidates:
            scores[doc_id] = self.score_document(doc_id, query)
            
        return scores


if __name__ == "__main__":
    from data.loader import load_arxiv_dataset
    docs = load_arxiv_dataset(sample_size=1000)
    from index.inverted_index import InvertedIndex
    index = InvertedIndex()
    index.build_index(docs, parallel=True, workers=4)
    bm25 = BM25Ranker(docs, index)
    query = input("Enter search query: ")
    scores = bm25.rank(query)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("BM25 Ranking Results:")
    for doc_id, score in ranked[:10]:
        print(f"{doc_id}: {score:.4f}")
