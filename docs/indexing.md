# Inverted Index and Related Data Structures

This section delves into the indexing mechanisms that underpin ArQiv's performance.

## Inverted Index

- **Purpose:**  
  The inverted index is the primary data structure that maps each unique term to a list of document identifiers and positions within the documents. This enables rapid retrieval for boolean and ranked searches.

- **Construction:**  
  Each document's combined title and content are tokenized, stopwords removed, and terms stemmed. Each term is then recorded along with its position.

## Trie Data Structure

- **Purpose:**  
  The trie supports fast prefix lookups, crucial for autocomplete and fuzzy search features.

- **Mechanism:**  
  Each node represents a character, and the trie is traversed to find words that match a given prefix.

## Bitmap Index

- **Purpose:**  
  A bitmap index accelerates boolean operations using vectorized operations in NumPy.

- **Mechanism:**  
  Each term's occurrence across documents is stored as a boolean array. Boolean operations (AND, OR) are then executed efficiently to produce candidate document sets.

These combined indexing techniques allow ArQiv to scale and retrieve results extremely fast.
