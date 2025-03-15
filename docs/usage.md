# Usage Instructions

ArQiv is engineered for simplicity without compromising on power. This guide explains how to use the search engine via its command-line interface (CLI).

## Launching ArQiv

From the project root directory, run:
```bash
python cli.py
```

Upon launch, you will see a banner followed by a menu of search options.

## Search Options

- **Basic Boolean Search:**  
  Quickly retrieves documents containing all query terms using the inverted index.
- **BM25 Ranking:**  
  Applies the BM25 algorithm to rank documents by relevance.
- **Fuzzy Search:**  
  Uses Levenshtein distance to perform approximate matching.
- **TF-IDF Ranking:**  
  Ranks documents based on vectorized term-frequency scores.
- **Fast Vector-Ranking:**  
  Utilizes a precomputed TF-IDF matrix with NearestNeighbors for rapid search.
- **Optional BERT Ranking:**  
  Employs SentenceTransformer models for deep semantic search, with an initialization prompt due to higher resource demands.

## Sample Workflow

1. **Start the Application:**  
   ArQiv loads documents, builds the index (or loads it from cache), and initializes rankers.
2. **Select a Search Mode:**  
   Choose an option from the menu.
3. **Enter Your Query:**  
   Provide your search terms.
4. **Review the Results:**  
   Search results are presented in detailed panels with highlighted key information.

The system caches recent queries to improve responsiveness on repeated searches.
