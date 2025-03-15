# Technical Details of ArQiv

Welcome to the technical deep-dive for ArQiv. In this document, we walk through the core components of the search engine.

## Data Loading and Preprocessing

- **Dataset Acquisition:**  
  ArQiv downloads the ArXiv metadata using Kaggle CLI (or falls back to curl if necessary).  
  The raw JSON lines are parsed into `Document` objects that store the ArXiv id, title, content, and additional metadata.

- **Preprocessing Pipeline:**  
  Text preprocessing involves:
  - **Tokenization:** Splitting text into words using regex.
  - **Stopword Removal:** Filtering out common English words via NLTK.
  - **Stemming:** Reducing words to their root using the Porter Stemmer.

  This pipeline ensures that the documents are normalized for indexing and retrieval.

## Indexing

- **Inverted Index:**  
  The core structure maps each term to its occurrences (positions) in each document.  
  For every document, the combined text (title + content) is preprocessed and then each term’s position is recorded.

- **Trie Data Structure:**  
  In addition to the inverted index, a Trie is built for fast prefix matching (useful for autocomplete and fuzzy queries).

- **Bitmap Index:**  
  Using NumPy arrays, a bitmap is built to quickly perform boolean operations (AND/OR) over documents.  
  This allows extremely fast candidate retrieval.

## Ranking Algorithms

ArQiv supports multiple ranking functions:
- **BM25 Ranking:**  
  A probabilistic ranking function that computes:
  ```
  score = Σ [ idf(term) × ((f × (k1 + 1)) / (f + k1 * (1 - b + b * (dl / avgdl))) ) ]
  ```
  where _f_ is term frequency, _dl_ is document length, and `k1`, `b` are tunable parameters.  
  IDF is computed as:
  ```
  idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)
  ```
  with _N_ being the total number of documents and _df_ the document frequency.

- **TF-IDF & Fast Vector Ranking:**  
  These rely on scikit‑learn’s `TfidfVectorizer` to convert documents into vectors. Cosine similarity is then computed between the query vector and document vectors.

- **BERT-based Ranking (Optional):**  
  Uses SentenceTransformer models to encode text into semantic embeddings.  
  Ranking is done via cosine similarity over high-dimensional embeddings.

## Parallel Processing

ArQiv builds the inverted index in parallel using Python’s `ProcessPoolExecutor`, breaking the dataset into chunks and merging partial indices at the end.

This document has provided an overview of the technical workings. In the next file, we will dive deeper into the math behind our BM25 ranking.
