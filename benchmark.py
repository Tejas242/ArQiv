import time
from data.loader import load_arxiv_dataset
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25Ranker

def benchmark_indexing():
    docs = load_arxiv_dataset(sample_size=1000)
    index = InvertedIndex()
    start = time.time()
    index.build_index(docs, parallel=True, workers=4)
    print(f"Indexing time (1000 docs, parallel): {time.time() - start:.2f} seconds")

def benchmark_bm25(query: str):
    docs = load_arxiv_dataset(sample_size=1000)
    index = InvertedIndex()
    index.build_index(docs, parallel=True, workers=4)
    bm25 = BM25Ranker(docs, index)
    start = time.time()
    scores = bm25.rank(query)
    print(f"BM25 ranking time: {time.time() - start:.2f} seconds")
    print("Top score:", max(scores.values(), default=0))

if __name__ == "__main__":
    benchmark_indexing()
    benchmark_bm25("deep learning")
