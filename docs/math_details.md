# Mathematical Foundations of Ranking Algorithms

This document thoroughly explains the mathematical principles behind ArQiv's ranking algorithms.

## BM25 Ranking

BM25 computes the relevance score of a document \( D \) for a query \( Q \) using:

\[
\text{score}(D, Q) = \sum_{term \in Q} \text{idf}(term) \times \frac{f(term, D) \times (k1 + 1)}{f(term, D) + k1 \times \left(1 - b + b \times \frac{dl}{avgdl}\right)}
\]

**Where:**

- \( f(term, D) \) = Frequency of the term in document \( D \)
- \( dl \) = Length of document \( D \)
- \( avgdl \) = Average document length in the corpus
- \( k1 \) and \( b \) are tunable parameters (commonly \( k1 \approx 1.5 \) and \( b \approx 0.75 \))
- \( \text{idf}(term) \) is computed as:

\[
\text{idf}(term) = \log\left(\frac{N - df + 0.5}{df + 0.5} + 1\right)
\]

with \( N \) being the total number of documents and \( df \) the document frequency of the term.

## TF-IDF and Cosine Similarity

TF-IDF transforms text into a vector space:

\[
\text{tf-idf}(t, D) = \text{tf}(t, D) \times \log\left(\frac{N}{df(t)}\right)
\]

Cosine similarity between the query vector \( Q \) and a document vector \( D \) is computed as:

\[
\text{similarity}(Q, D) = \frac{Q \cdot D}{\|Q\| \times \|D\|}
\]

These approaches allow efficient, vectorized scoring of document relevance.

---

The above formulas provide the theoretical basis for our ranking strategies, ensuring that rare terms contribute significantly to the score while normalizing for document length.
