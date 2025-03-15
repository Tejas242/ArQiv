# Performance Benchmarks for ArQiv

This document details the results of performance benchmarks run on ArQiv.

## Indexing Performance

- Built inverted index on 1,000 documents in approximately **0.79 seconds** (using parallel indexing with 4 workers).

## BM25 Ranking

- Average BM25 response time for query "deep learning" is approximately **0.0045 milliseconds**.

## Future Improvements

- Consider moving heavy operations (such as BERT ranking) to GPU-enabled environments.
- Explore further multiprocess optimizations and caching strategies.

*Note: These benchmarks were run on a HP Notebook (CPU: Ryzen 3, RAM: 8GB) machine.*
