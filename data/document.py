from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class Document:
    """
    A class representing a document in the ArQiv search engine.
    
    Attributes:
        doc_id: Unique identifier for the document (ArXiv ID)
        title: Document title
        content: Document content (typically the abstract)
        metadata: Additional document metadata (authors, categories, etc.)
    """
    doc_id: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
