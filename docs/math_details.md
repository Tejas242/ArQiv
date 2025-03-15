# Mathematical Foundations

This document details the formulas that power ArQiv's ranking mechanisms in full detail.

## BM25 Scoring

For a document $$D$$ and a query $$Q$$, the BM25 score is computed as:
$$
\text{score}(D,Q) = \sum_{term \in Q} \text{idf}(term) \cdot \frac{f(term, D) \cdot (k_1 + 1)}{f(term, D) + k_1 \cdot \Bigl(1 - b + b \cdot \frac{dl}{avgdl}\Bigr)}
$$

where:
- $$f(term, D)$$ is the frequency of the term in document $$D$$.
- $$dl$$ denotes the length (number of terms) in document $$D$$.
- $$avgdl$$ represents the average document length across the whole corpus.
- $$k_1$$ is a tuning parameter (typically around 1.5) that controls the scaling of term frequency.
- $$b$$ is a document length normalization parameter (typically around 0.75).
- The inverse document frequency (IDF) is given by:
$$
\text{idf}(term) = \log\!\Biggl(\frac{N - df + 0.5}{df + 0.5} + 1\Biggr)
$$
with:
  - $$N$$ = total number of documents,
  - $$df$$ = number of documents containing the term.

These components ensure that rare terms (high IDF) are emphasized, and document length is normalized.

## TF-IDF and Cosine Similarity

The TF-IDF weight for a term $$t$$ in a document $$D$$ is calculated as:
$$
\text{tf-idf}(t, D) = tf(t, D) \times \log\!\Biggl(\frac{N}{df(t)}\Biggr)
$$
where:
- $$tf(t, D)$$ is the frequency of term $$t$$ in document $$D$$,
- $$df(t)$$ is the document frequency of $$t$$,
- $$N$$ is the total number of documents.

Cosine similarity between the query vector $$Q$$ and a document vector $$D$$ is defined as:
$$
\text{similarity}(Q, D) = \frac{Q \cdot D}{\|Q\| \, \|D\|}
$$

This measures the cosine of the angle between the two vectors, providing a normalized measure of similarity.

---

**Detailed Explanation:**

- **Term Frequency (TF):** Measures the occurrence of a term in a document.
- **Inverse Document Frequency (IDF):** Dampens the weight of frequently occurring terms across documents.
- **Normalization:** The BM25 formula normalizes for document length to avoid bias toward longer documents.
- **Cosine Similarity:** Focuses on the orientation of vectors rather than their magnitude, giving a robust similarity score even if documents vary in length.

These mathematical foundations ensure that the ranking algorithms in ArQiv deliver balanced and relevant search results.
