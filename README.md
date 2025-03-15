# 🚀 ArQiv

<div style="background-color:#f0f8ff; padding:20px; border-radius:10px; margin-bottom:20px;">
    <h1 style="color:#2e8b57; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin:0;">Welcome to ArQiv!</h1>
    <p style="font-size:16px; margin:10px 0 0 0;">
        ArQiv is a state-of-the-art search engine, meticulously designed for the ArXiv research corpus. By leveraging powerful data structures such as inverted indexes, tries, and bitmap indexes alongside advanced ranking algorithms like BM25, TF-IDF, Fast Vector Ranking, and optional BERT-based semantic search, ArQiv delivers ultra-fast, precise, and scalable search results that empower researchers and developers alike.
    </p>
</div>

## ✨ Key Features

- **Optimized Data Structures:**
  - 🔍 **Inverted Index:** Rapid term lookups with positional data.
  - 📝 **Trie (Prefix Tree):** Instant autocomplete and fuzzy matching.
  - ⚡ **Bitmap Index:** Ultra-fast Boolean operations leveraging vectorization.
- **Advanced Ranking Algorithms:**
  - 🏆 **BM25 Ranking:** Precise probabilistic relevance scoring.
  - 📊 **TF-IDF Ranking:** Robust vectorized similarity computation.
  - ⚙️ **Fast Vector Ranking:** Real-time ranking via NearestNeighbors.
  - 🤖 **Optional BERT Ranking:** Deep semantic ranking using SentenceTransformer.
- **Intuitive and Interactive TUI:**
  - Colorful, dynamic CLI with rich banners and detailed result panels.
  - In-memory query caching for near-instant feedback.
- **Scalable & Modular:**
  - Parallel processing for seamless index building.
  - Extensible framework that adapts to evolving research needs.

## 📂 Project Structure

- **data/**
  - `document.py`: Defines the Document data class.
  - `preprocessing.py`: Implements tokenization, stopword removal, and stemming.
  - `loader.py`: Loads and caches the ArXiv dataset with automatic download support.
- **index/**
  - `inverted_index.py`: Core indexing module with positional data and bitmap integration.
  - `trie.py`: Trie for fast prefix matching.
  - `bitmap_index.py`: Bitmap index for rapid Boolean search.
- **ranking**
  - `bm25.py`: BM25 ranking algorithm.
  - `tfidf.py`: TF-IDF ranking using Scikit‑Learn.
  - `fast_vector_ranker.py`: Fast vector-space ranking with NearestNeighbors.
  - `bert_ranker.py`: Optional BERT-based ranking.
- **search/**
  - `fuzzy_search.py`: Implements fuzzy matching with Levenshtein distance.
- **cli.py**: Interactive TUI for end users.
- **analysis/**
  - Jupyter notebooks for in-depth data analysis and benchmarking.

## 🔧 Setup and Installation

1. **Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd Fun/ArQiv
   ```

2. **Create and Activate a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install datasets nltk rich numpy scikit-learn kaggle sentence-transformers
   ```

4. **Download NLTK Resources:**
   Open a Python shell and run:
   ```python
   import nltk
   nltk.download('stopwords')
   ```
   *(After your first run, you may comment out the download line in `preprocessing.py`.)*

5. **Configure the Kaggle API:**
   - Install the Kaggle CLI with:
     ```bash
     pip install kaggle
     ```
   - Follow the [Kaggle CLI setup instructions](https://github.com/Kaggle/kaggle-api) to configure your API credentials.
   - The application will automatically download the ArXiv dataset if it's missing.

## 🚀 Usage

Launch the interactive search engine by running:
```bash
python cli.py
```

### Search Options

- **Basic Boolean Search:** Ultra-fast lookup using the inverted index.
- **BM25 Ranking:** Accurate scoring based on probabilistic relevance.
- **Fuzzy Search:** Approximate matching using Levenshtein distance.
- **TF-IDF Ranking:** Rapid vectorized similarity calculations.
- **Fast Vector-Ranking:** Lightweight ranking using precomputed vectors.
- **BERT Ranking (Optional):**
  - Prompts initialization for resource-heavy semantic ranking.

## 📸 Screenshots

<div style="border:2px dashed #ccc; padding:10px; text-align:center; border-radius:8px;">
    <p style="color:#888; font-style:italic;">[Screenshots will be added here shortly...]</p>
    <img src="path/to/screenshot_placeholder.png" alt="Screenshot placeholder" style="max-width:100%; opacity:0.5;">
</div>

## 📑 Sample Workflow

1. **Start the Application:**  
   ArQiv loads documents, builds (or loads) the search index, and initializes rankers.
2. **View the Banner and Menu:**  
   A dynamic, visually appealing banner and menu are displayed.
3. **Enter Your Query:**  
   Select a search mode and input your query. Detailed results appear in structured panels with document ID, title, authors, and highlighted content.

## 🛠 Troubleshooting

- **Dataset Download Issues:**  
  Verify that the Kaggle CLI is installed and properly configured. The application will fallback to using `curl` when necessary.
- **Performance on CPU:**  
  For best performance on CPU-only systems, use BM25, TF-IDF, or Fast Vector-Ranking. BERT Ranking is optional and resource intensive.
- **Index Rebuilding:**  
  If outdated index caches cause issues, they will automatically be rebuilt.

## 🤝 Contributions

We welcome contributions! Whether you’re fixing bugs, adding features, or improving performance, please adhere to best practices and update relevant documentation.

## 📜 License

[MIT License](LICENSE)
