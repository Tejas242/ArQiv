from typing import List, Dict
from data.document import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


class FastVectorRanker:
    """
    A fast vector-space ranker that precomputes TF-IDF vectors and uses
    NearestNeighbors for efficient cosine similarity search.
    
    This algorithm is lightweight and typically returns results in <5ms,
    making it ideal for real-time search applications.
    
    Attributes:
        documents: List of documents to rank
        doc_ids: List of document IDs
        vectorizer: TF-IDF vectorizer from scikit-learn
        tfidf_matrix: Precomputed TF-IDF matrix
        nn_model: NearestNeighbors model for fast similarity search
    """
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]
        
        # Combine title and content for more comprehensive ranking
        corpus = [doc.title + " " + doc.content for doc in documents]
        
        # Create TF-IDF representation
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # Initialize NearestNeighbors for fast cosine similarity
        self.nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn_model.fit(self.tfidf_matrix)
    
    def rank(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        Rank documents using nearest neighbor search with cosine distance.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary mapping document IDs to similarity scores
        """
        # Transform query to TF-IDF vector space
        query_vec = self.vectorizer.transform([query])
        
        # Find k nearest neighbors by cosine distance
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=min(top_k, len(self.doc_ids)))
        
        # Convert distance to similarity (cosine similarity = 1 - cosine distance)
        scores = 1 - distances.flatten()
        
        # Create ranking dictionary
        ranking = {self.doc_ids[idx]: float(score) for idx, score in zip(indices.flatten(), scores)}
        return ranking
