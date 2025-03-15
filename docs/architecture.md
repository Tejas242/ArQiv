# ArQiv System Architecture

ArQiv features a modular, layered architecture that ensures high performance and extensibility.

## High-Level Components

- **User Interface**
  - CLI (built with Rich)
  - Web UI (Streamlit)

- **Ranking Module**
  - BM25, TF-IDF, Fast Vector, and optional BERT rankers

- **Indexing Module**
  - **Inverted Index:** Core structure mapping terms to documents.
  - **Trie:** Enables efficient autocomplete and fuzzy search.
  - **Bitmap Index:** Facilitates rapid Boolean operations.

- **Data Module**
  - **Document Model:** Standardizes metadata.
  - **Preprocessing:** Tokenizes, cleans, and stems input text.
  - **Loader:** Manages dataset download and caching.

## Data Flow

1. **Data Ingestion:**  
   Raw metadata is parsed into Document objects.
2. **Preprocessing:**  
   Text is normalized for indexing.
3. **Index Construction:**  
   Inverted, trie, and bitmap indexes are built.
4. **Query Processing:**  
   Input queries are preprocessed, candidate documents are fetched, and ranking algorithms compute relevance.
5. **Result Presentation:**  
   The UI displays enhanced search results with highlighted snippets.

This architecture allows ArQiv to scale while maintaining responsiveness and accuracy.
