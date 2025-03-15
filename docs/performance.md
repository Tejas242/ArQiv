# ArQiv Performance Benchmarks

Evaluate ArQiv’s speed and efficiency with these benchmarks measured on a Ryzen 3 CPU with 8GB RAM.

| Task                        | Performance         |
| --------------------------- | ------------------- |
| Indexing 1,000 documents    | ~0.8 seconds        |
| Boolean Search              | < 5 ms              |
| BM25 Ranking                | 50 – 100 ms         |
| TF-IDF Ranking              | < 5 ms              |
| Fast Vector Ranking         | < 5 ms              |
| BERT Ranking (optional)     | ~200 ms             |

_Notes: Optimizations like parallel processing and in-memory caching ensure rapid turnaround even on modest hardware._
