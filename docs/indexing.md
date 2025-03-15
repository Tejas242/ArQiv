# Inverted Index & Data Structures

ArQiv’s performance relies on a suite of optimized data structures.

## Inverted Index

- **Purpose:**  
  Maps each unique term to its document identifiers and positions, enabling fast term lookup.

- **Process:**  
  Each document’s text (title + content) is tokenized, stopwords are removed, and terms are stemmed before indexing.

## Trie (Prefix Tree)

- **Usage:**  
  Powers autocomplete and fuzzy search by rapidly matching term prefixes.

## Bitmap Index

- **Function:**  
  Converts postings into boolean arrays for lightning-fast Boolean operations (AND, OR), independent of posting list lengths.

Combined, these structures allow ArQiv to deliver sub-millisecond query responses.
