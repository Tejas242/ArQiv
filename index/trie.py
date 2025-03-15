from typing import Dict, List, Optional


class TrieNode:
    """
    Node in a Trie (prefix tree) data structure.
    
    Attributes:
        children: Dictionary mapping characters to child nodes
        is_end_of_word: Flag indicating if this node completes a word
        postings: Dictionary mapping document IDs to positions
    """
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end_of_word: bool = False
        self.postings: Dict[str, List[int]] = {}  # doc_id -> positions


class Trie:
    """
    Trie data structure for fast string operations like prefix matching.
    
    This structure significantly speeds up autocomplete and fuzzy search operations
    by providing efficient prefix lookups.
    """
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word: str, doc_id: str, position: int) -> None:
        """
        Insert a word into the Trie and record document and position.
        
        Args:
            word: The word to insert
            doc_id: Document ID containing the word
            position: Position of the word in the document
        
        Time complexity: O(len(word))
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        node.is_end_of_word = True
        if doc_id not in node.postings:
            node.postings[doc_id] = []
        node.postings[doc_id].append(position)
    
    def search(self, word: str) -> Optional[Dict[str, List[int]]]:
        """
        Search for exact word match in the Trie.
        
        Args:
            word: The word to search for
        
        Returns:
            Dictionary mapping document IDs to positions, or None if not found
        
        Time complexity: O(len(word))
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
            
        return node.postings if node.is_end_of_word else None
    
    def starts_with(self, prefix: str) -> List[str]:
        """
        Find all words that start with the given prefix.
        
        Args:
            prefix: The prefix to search for
        
        Returns:
            List of words with the given prefix
        
        Time complexity: O(len(prefix) + number of matching words)
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
            
        words = []
        self._collect_words(node, prefix, words)
        return words
    
    def _collect_words(self, node: TrieNode, prefix: str, words: List[str]) -> None:
        """
        Helper method to recursively collect words from Trie nodes.
        
        Args:
            node: Current Trie node
            prefix: Current prefix string
            words: List to collect words into
        """
        if node.is_end_of_word:
            words.append(prefix)
            
        for char, child in node.children.items():
            self._collect_words(child, prefix + char, words)
