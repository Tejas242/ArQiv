<div align="center">
  <h1>🚀 ArQiv Search Engine</h1>
  <p><i>High-performance semantic search for ArXiv research papers</i></p>
  <img src="https://img.shields.io/badge/ArQiv-v1.0.0-2e8b57.svg?style=for-the-badge" alt="ArQiv Version" />
  
  <p>
    <a href="#-key-features">Features</a> •
    <a href="#-getting-started">Getting Started</a> •
    <a href="#-usage">Usage</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-benchmarks">Benchmarks</a>
  </p>
  
  <!-- Technology badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
    <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy" />
    <img src="https://img.shields.io/badge/SciKit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn" />
    <img src="https://img.shields.io/badge/NLTK-222222?style=for-the-badge" alt="NLTK" />
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
    <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly" />
  </p>
</div>

---

## 📖 Overview

**ArQiv** is a state-of-the-art search engine designed specifically for the ArXiv research corpus. It combines multiple indexing strategies and ranking algorithms to deliver lightning-fast, relevant results that help researchers discover papers more efficiently.

<div align="center">
  <img src="https://via.placeholder.com/800x400?text=ArQiv+Screenshot" alt="ArQiv Interface" width="80%" />
</div>

## ✨ Key Features

<table>
  <tr>
    <td width="50%">
      <h3>🔍 Optimized Data Structures</h3>
      <ul>
        <li><b>Inverted Index:</b> Rapid term lookups with positional data</li>
        <li><b>Trie (Prefix Tree):</b> Instant autocomplete and fuzzy matching</li>
        <li><b>Bitmap Index:</b> Ultra-fast Boolean operations with vectorization</li>
      </ul>
    </td>
    <td width="50%">
      <h3>📊 Advanced Ranking Algorithms</h3>
      <ul>
        <li><b>BM25 Ranking:</b> Precise probabilistic relevance scoring</li>
        <li><b>TF-IDF Ranking:</b> Robust vectorized similarity computation</li>
        <li><b>Fast Vector Ranking:</b> Real-time ranking via NearestNeighbors</li>
        <li><b>Optional BERT Ranking:</b> Deep semantic ranking with transformers</li>
      </ul>
    </td>
  </tr>
  <tr>
    <td>
      <h3>🖥️ Multiple Interfaces</h3>
      <ul>
        <li><b>Rich CLI:</b> Colorful terminal interface with detailed results</li>
        <li><b>Streamlit Web App:</b> Interactive web UI with visualizations</li>
        <li><b>In-memory Query Caching:</b> Near-instant response on repeated queries</li>
      </ul>
    </td>
    <td>
      <h3>⚡ Performance & Scalability</h3>
      <ul>
        <li><b>Parallel Processing:</b> Multi-core indexing for large datasets</li>
        <li><b>Modular Design:</b> Extensible architecture for new features</li>
        <li><b>Memory Efficient:</b> Optimized for performance on standard hardware</li>
      </ul>
    </td>
  </tr>
</table>

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- 4GB+ RAM (8GB recommended for full dataset)
- Internet connection for initial dataset download

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/arqiv.git
   cd arqiv
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK resources (one-time):**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## 📋 Usage

### Command Line Interface

Launch the rich CLI interface:

```bash
python cli.py
```

### Streamlit Web Interface

Start the interactive web application:

```bash
cd streamlit
streamlit run streamlit_app.py
```

<div align="center">
  <img src="https://via.placeholder.com/800x120?text=ArQiv+CLI+Demo" alt="CLI Demo" width="80%" />
</div>

## 🏗️ Architecture

ArQiv employs a layered architecture with the following components:

```
┌─────────────────────────────────┐
│          User Interfaces        │
│    CLI Interface  │  Streamlit  │
├─────────────────────────────────┤
│         Ranking Algorithms      │
│  BM25  │ TF-IDF │ Vector │ BERT │
├─────────────────────────────────┤
│          Index Structures       │
│ Inverted Index │ Trie │ Bitmap  │
├─────────────────────────────────┤
│             Data Layer          │
│ Document Model │ Preprocessing  │
└─────────────────────────────────┘
```

## 📊 Benchmarks

| Task                  | Performance |
| --------------------- | ----------- |
| Index 1,000 documents | 0.8 seconds |
| Boolean search        | < 5ms       |
| BM25 ranking          | ~50-100ms   |
| TF-IDF ranking        | < 5ms       |
| Fast Vector ranking   | < 5ms       |
| BERT ranking          | ~200ms      |

_Measurements on Ryzen 3 CPU with 8GB RAM_

## 📁 Project Structure

<details>
<summary>Click to expand directory structure</summary>

```
arqiv/
├── data/               # Data handling components
│   ├── document.py     # Document model
│   ├── preprocessing.py # Text processing utilities
│   └── loader.py       # Dataset loading
├── index/              # Indexing structures
│   ├── inverted_index.py # Main indexing engine
│   ├── trie.py         # Prefix tree for autocomplete
│   └── bitmap_index.py # Bitmap for fast boolean ops
├── ranking/            # Ranking algorithms
│   ├── bm25.py         # BM25 implementation
│   ├── tfidf.py        # TF-IDF with scikit-learn
│   ├── fast_vector_ranker.py # Vector-based ranking
│   └── bert_ranker.py  # Neural ranking with BERT
├── search/             # Search functionalities
│   └── fuzzy_search.py # Approximate string matching
├── streamlit/          # Web interface
│   └── streamlit_app.py # Streamlit application
├── docs/               # Documentation
├── tests/              # Test suite
├── cli.py              # Command-line interface
└── README.md           # This file
```

</details>

## 🔧 Troubleshooting

<details>
<summary>Common issues and solutions</summary>

### Dataset Download Issues

- Ensure Kaggle API credentials are set up correctly
- For manual download: Place the arxiv-metadata-oai-snapshot.json in the data/ directory

### Performance Problems

- For slow search: Try using BM25 or TF-IDF ranking for faster results
- If indexing is slow: Increase the number of worker processes in the parallel option

### Memory Usage

- If experiencing memory issues: Reduce sample_size parameter in load_arxiv_dataset()
</details>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p>
    <sub>Built with ⚡ by Tejas Mahajan</sub>
  </p>
</div>
