from typing import List, Dict
from data.document import Document
import numpy as np
from sentence_transformers import SentenceTransformer, util


class BERTRanker:
    """
    Neural ranking algorithm using SentenceTransformer models.
    
    Computes semantic embeddings for documents and queries, then ranks
    using cosine similarity in the embedding space. This approach captures
    semantic relationships beyond simple keyword matching.
    
    Attributes:
        documents: List of documents to rank
        doc_ids: List of document IDs
        model: SentenceTransformer model
        embeddings: Precomputed document embeddings
    """
    def __init__(self, documents: List[Document], model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize BERT ranker with documents and a transformer model.
        
        Args:
            documents: List of documents to embed and rank
            model_name: Name of the SentenceTransformer model to use
        """
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]
        
        # Combine title and content for each document
        corpus = [doc.title + " " + doc.content for doc in documents]
        
        # Initialize model with CPU device explicitly for compatibility
        self.model = SentenceTransformer(model_name, device="cpu")
        
        # Precompute document embeddings
        print("Computing document embeddings... (this may take a while)")
        self.embeddings = self.model.encode(corpus, convert_to_tensor=True, 
                                           show_progress_bar=True)
        print("Embeddings computed successfully")

    def rank(self, query: str, top_k: int = 10) -> Dict[str, float]:
        """
        Rank documents against the query using neural embeddings.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            Dictionary mapping document IDs to similarity scores
        """
        # Encode query to same embedding space as documents
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        
        # Compute cosine similarities between query and all documents
        cosine_scores = util.cos_sim(query_embedding, self.embeddings)[0]
        
        # Get indices of top_k highest scores
        top_k = min(top_k, len(self.doc_ids))
        top_results = np.argpartition(-cosine_scores.cpu().numpy(), range(top_k))[:top_k]
        
        # Create ranking dictionary
        ranking = {self.doc_ids[i]: float(cosine_scores[i].cpu().numpy()) 
                  for i in top_results}
        return ranking
