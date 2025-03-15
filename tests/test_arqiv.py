import pytest
from data.document import Document
from data.preprocessing import tokenize, remove_stopwords, stem_tokens, preprocess
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25Ranker
from ranking.tfidf import TFIDFRanker
from ranking.fast_vector_ranker import FastVectorRanker
from search.fuzzy_search import levenshtein_distance, fuzzy_search

# Sample document used across multiple tests
SAMPLE_DOC = Document(
    doc_id="doc1", 
    title="Deep Learning Advances", 
    content="Deep learning transforms research in academia and industry.",
    metadata={"authors": "Author A, Author B", "categories": "cs.LG"}
)

# Additional sample document for multi-document tests
SAMPLE_DOC_2 = Document(
    doc_id="doc2", 
    title="Machine Learning Basics", 
    content="Machine learning includes supervised and unsupervised techniques.",
    metadata={"authors": "Author C", "categories": "cs.AI"}
)


# ---------------------- Document Model Tests ----------------------
def test_document_fields():
    """
    Test that Document fields are correctly assigned.
    """
    doc = SAMPLE_DOC
    assert doc.doc_id == "doc1"
    assert "Deep Learning" in doc.title
    assert "transforms" in doc.content
    assert "authors" in doc.metadata
    assert isinstance(doc.metadata, dict)


# ---------------------- Preprocessing Pipeline Tests ----------------------
def test_tokenize():
    """
    Test tokenize returns only lowercase words.
    """
    text = "Hello World! This is a TEST."
    tokens = tokenize(text)
    assert all(token.islower() for token in tokens)
    assert "hello" in tokens and "world" in tokens

def test_remove_stopwords():
    """
    Test that common stopwords are removed.
    """
    tokens = ["this", "is", "a", "test", "of", "removal"]
    filtered = remove_stopwords(tokens)
    for stopword in ["this", "is", "a", "of"]:
        assert stopword not in filtered

def test_stem_tokens():
    """
    Test that stemming produces expected results.
    """
    tokens = ["running", "jumps", "easily"]
    stemmed = stem_tokens(tokens)
    # Expect stems; for example, "run", "jump" (actual outputs may differ)
    assert any("run" in token for token in stemmed)
    assert any("jump" in token for token in stemmed)

def test_preprocess():
    """
    Test the full preprocessing pipeline.
    """
    text = "This is a SIMPLE Test, for Preprocessing!!!"
    processed = preprocess(text)
    assert isinstance(processed, list)
    for token in processed:
        assert token == token.lower()


# ---------------------- Inverted Index Tests ----------------------
def test_index_document_and_search():
    """
    Test that a document indexed can be found via token search.
    """
    index = InvertedIndex()
    index.index_document(SAMPLE_DOC)
    # Search for a token from title; check case-insensitivity.
    result = index.search("deep")
    assert "doc1" in result

def test_index_multiple_documents():
    """
    Test indexing of multiple documents and candidate retrieval.
    """
    index = InvertedIndex()
    index.index_document(SAMPLE_DOC)
    index.index_document(SAMPLE_DOC_2)
    candidates = index.get_candidate_docs("learning")
    # Both documents mention 'learning'
    assert "doc1" in candidates
    assert "doc2" in candidates

# ---------------------- BM25 Ranker Tests ----------------------
def test_bm25_ranking_positive_score():
    """
    Test BM25 ranking returns a positive score for a matching query.
    """
    docs = [SAMPLE_DOC, SAMPLE_DOC_2]
    index = InvertedIndex()
    for d in docs:
        index.index_document(d)
    bm25 = BM25Ranker(docs, index)
    scores = bm25.rank("deep learning")
    # At least one document should have a positive score
    assert any(score > 0 for score in scores.values())

def test_bm25_edge_empty_query():
    """
    Test BM25 ranking with empty query returns empty scores.
    """
    docs = [SAMPLE_DOC]
    index = InvertedIndex()
    index.index_document(SAMPLE_DOC)
    bm25 = BM25Ranker(docs, index)
    scores = bm25.rank("")
    assert scores == {}

# ---------------------- TF-IDF Ranker Tests ----------------------
def test_tfidf_ranking():
    """
    Test TF-IDF ranking produces non-zero similarity scores.
    """
    docs = [SAMPLE_DOC, SAMPLE_DOC_2]
    tfidf_ranker = TFIDFRanker(docs)
    ranking = tfidf_ranker.rank("machine learning")
    # Ensure that documents with matching content earn a score
    assert any(score > 0 for score in ranking.values())

# ---------------------- Fast Vector Ranker Tests ----------------------
def test_fast_vector_ranker():
    """
    Test Fast Vector Ranker returns results with high cosine similarity.
    """
    docs = [SAMPLE_DOC, SAMPLE_DOC_2]
    fast_vector_ranker = FastVectorRanker(docs)
    ranking = fast_vector_ranker.rank("academia")
    assert isinstance(ranking, dict)
    # At least one document should have a similarity score > 0
    assert any(score > 0 for score in ranking.values())

# ---------------------- Fuzzy Search and Levenshtein Distance Tests ----------------------
def test_levenshtein_distance():
    """
    Test that the Levenshtein function computes expected edit distance.
    """
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("flaw", "lawn") == 2

def test_fuzzy_search():
    """
    Test fuzzy search returns document IDs when matching tokens within threshold.
    """
    index = InvertedIndex()
    index.index_document(SAMPLE_DOC)
    # Introduce a typo in query token: "deap" should match "deep" with max_distance=1.
    results = fuzzy_search("deap", index, max_distance=1)
    assert "doc1" in results

if __name__ == "__main__":
    pytest.main()
