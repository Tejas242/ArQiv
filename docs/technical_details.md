# Technical Deep-Dive into ArQiv

Discover the inner workings of ArQiv, from data ingestion to state-of-the-art ranking algorithms.

## Data Ingestion and Preprocessing

- **Acquisition:**  
  ArQiv downloads ArXiv metadata using Kaggle CLI (with fallback to curl) and converts JSONL records into standardized Document objects.

- **Preprocessing Pipeline:**  
  The text is tokenized, cleansed of stopwords (via NLTK), and stemmed using PorterStemmer for normalization.

## Indexing Strategies

- **Inverted Index:**  
  Maps individual terms to their positions in each document. Essential for rapid boolean and ranked queries.

- **Trie and Bitmap Indexes:**  
  - **Trie:** Enables fast autocomplete and fuzzy matching.  
  - **Bitmap:** Uses vectorized Boolean operations for sub-millisecond candidate retrieval.

## Ranking Algorithms

ArQiv supports several ranking functions:
- **BM25:** Probabilistic scoring with document length normalization.
- **TF-IDF:** Vector space models with cosine similarity.
- **Fast Vector Ranking:** NearestNeighbors search for real-time responses.
- **BERT (Optional):** Transformer-based semantic ranking (resource intensive).

This document underpins the design choices and math that drive ArQiv.
