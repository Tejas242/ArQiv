# System Architecture

This document explains the overall architecture of ArQiv, detailing its modular design and data flow.

## High-Level Modules

- **Data Module:**  
  - *Document Model:* Defines a standardized structure for documents.
  - *Preprocessing:* Tokenizes text, removes stopwords, and applies stemming.
  - *Loader:* Handles dataset downloading, caching, and JSON parsing.

- **Indexing Module:**  
  - *Inverted Index:* Maps terms to document IDs and positions to support efficient search.
  - *Trie:* Enables fast prefix matching and fuzzy search capabilities.
  - *Bitmap Index:* Uses NumPy arrays to perform rapid boolean operations for candidate retrieval.

- **Ranking Module:**  
  - *BM25 Ranker:* Implements a probabilistic relevance scoring function.
  - *TF-IDF Ranker:* Leverages vectorization and cosine similarity.
  - *Fast Vector Ranker:* Uses NearestNeighbors for ultra-fast scoring.
  - *BERT Ranker (Optional):* Provides semantic search through deep learning embeddings.

- **User Interface Module:**  
  - The CLI utilizes the Rich library to provide an attractive and responsive interface for users.

## Data Flow Overview

1. **Data Ingestion:**  
   Raw metadata is downloaded and parsed into `Document` objects.

2. **Preprocessing:**  
   Each document is tokenized, normalized, and indexed.

3. **Index Construction:**  
   Documents are indexed using multiple structures for fast search and ranking.

4. **Query Processing:**  
   User queries are preprocessed, and results are obtained either by direct lookup or via ranking algorithms.

5. **Result Presentation:**  
   The CLI displays results interactively with highlighting and detailed metadata.

This modular architecture ensures both flexibility in ranking and robust performance.
