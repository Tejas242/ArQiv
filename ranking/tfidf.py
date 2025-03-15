from typing import List, Dict
from data.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFRanker:
    """
    TF-IDF ranking algorithm using scikit-learn's vectorizer.
    
    Uses cosine similarity between query vector and document vectors
    for efficient similarity-based ranking.
    
    Attributes:
        documents: List of documents to rank
        doc_ids: List of document IDs in the same order as the TF-IDF matrix
        vectorizer: scikit-learn's TF-IDF vectorizer
        tfidf_matrix: Precomputed TF-IDF matrix for documents
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]
        
        # Combine title and content for better ranking
        corpus = [doc.title + " " + doc.content for doc in documents]
        
        # Initialize and fit the vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def rank(self, query: str) -> Dict[str, float]:
        """
        Rank documents against the query using TF-IDF and cosine similarity.
        
        Args:
            query: Query string to rank documents against
            
        Returns:
            Dictionary mapping document IDs to similarity scores
        """
        query_vec = self.vectorizer.transform([query])
        
        # Compute cosine similarity between query vector and all document vectors
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Return only documents with non-zero scores (relevant documents)
        ranking = {self.doc_ids[i]: float(scores[i]) for i in np.where(scores > 0)[0]}
        return ranking
