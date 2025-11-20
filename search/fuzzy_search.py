from index.inverted_index import InvertedIndex
from data.preprocessing import preprocess

def levenshtein_distance(s1: str, s2: str) -> int:
    # Compute the Levenshtein edit distance between s1 and s2.
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def fuzzy_search(query: str, index: InvertedIndex, max_distance: int = 1):
    """
    Performs fuzzy search on the inverted index using Levenshtein distance.
    Returns a set of document IDs that match query tokens within the max_distance threshold.
    """
    query_tokens = preprocess(query)
    matched_docs = set()
    # Iterate over every term in the index.
    for term in list(index.index.keys()):
        for q in query_tokens:
            if levenshtein_distance(q, term) <= max_distance:
                matched_docs.update(index.index[term].keys())
                break  # Found a match for this term; move on.
    return matched_docs
