import pytest
from data.document import Document
from index.inverted_index import InvertedIndex

def test_index_document():
    # Create a simple document for testing
    doc = Document(doc_id="test1", title="Test Title", content="Test content for indexing.")
    index = InvertedIndex()
    index.index_document(doc)
    assert "test" in index.index  # basic check for token "test"
    # ...additional tests...

if __name__ == "__main__":
    pytest.main()