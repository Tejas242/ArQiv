from typing import Dict, List, Set
import numpy as np


class BitmapIndex:
    """
    Bitmap index for ultra-fast boolean operations (AND, OR, NOT).
    
    Uses NumPy arrays for efficient bitmap operations, enabling constant time
    boolean queries regardless of the number of documents or terms.
    
    Attributes:
        num_docs: Total number of documents in the collection
        term_to_bitmap: Maps terms to their bitmap representation
        doc_id_to_idx: Maps document IDs to their position in the bitmap
        idx_to_doc_id: Maps bitmap positions back to document IDs
    """
    def __init__(self, num_docs: int):
        self.num_docs = num_docs
        self.term_to_bitmap: Dict[str, np.ndarray] = {}
        self.doc_id_to_idx: Dict[str, int] = {}
        self.idx_to_doc_id: Dict[int, str] = {}
        
    def add_document(self, doc_id: str, idx: int) -> None:
        """
        Register a document ID with its bitmap index position.
        
        Args:
            doc_id: Document identifier
            idx: Index position in bitmap arrays
        """
        self.doc_id_to_idx[doc_id] = idx
        self.idx_to_doc_id[idx] = doc_id
        
    def add_term(self, term: str, doc_ids: List[str]) -> None:
        """
        Create a bitmap for a term, setting bits for documents that contain it.
        
        This enables O(1) set operations regardless of posting list size.
        
        Args:
            term: Term to add
            doc_ids: List of document IDs containing the term
        """
        bitmap = np.zeros(self.num_docs, dtype=bool)
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_idx:
                bitmap[self.doc_id_to_idx[doc_id]] = True
        self.term_to_bitmap[term] = bitmap
        
    def boolean_and(self, terms: List[str]) -> Set[str]:
        """
        Perform an AND operation on terms using bitmap operations.
        
        Args:
            terms: List of terms to AND together
            
        Returns:
            Set of document IDs containing all terms
            
        Time complexity: O(num_docs) regardless of result size
        """
        if not terms or any(term not in self.term_to_bitmap for term in terms):
            return set()
            
        result = self.term_to_bitmap[terms[0]].copy()
        for term in terms[1:]:
            result &= self.term_to_bitmap[term]
            
        doc_indices = np.where(result)[0]
        return {self.idx_to_doc_id[idx] for idx in doc_indices}
        
    def boolean_or(self, terms: List[str]) -> Set[str]:
        """
        Perform an OR operation on terms using bitmap operations.
        
        Args:
            terms: List of terms to OR together
            
        Returns:
            Set of document IDs containing any of the terms
            
        Time complexity: O(num_docs) regardless of result size
        """
        if not terms:
            return set()
            
        valid_terms = [term for term in terms if term in self.term_to_bitmap]
        if not valid_terms:
            return set()
            
        result = self.term_to_bitmap[valid_terms[0]].copy()
        for term in valid_terms[1:]:
            result |= self.term_to_bitmap[term]
            
        doc_indices = np.where(result)[0]
        return {self.idx_to_doc_id[idx] for idx in doc_indices}
