import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords if necessary - run once and comment out later.
import nltk
nltk.download('stopwords')

# Precompile regex for tokenization - more efficient for multiple calls
TOKEN_PATTERN = re.compile(r'\b\w+\b')

# Use NLTK's English stopwords
STOPWORDS = set(stopwords.words('english'))

# Initialize stemmer once
ps = PorterStemmer()


def tokenize(text: str) -> list:
    """
    Tokenize the input text into lowercase words.
    
    Args:
        text: Input text string to tokenize
        
    Returns:
        List of lowercase tokens
    """
    return TOKEN_PATTERN.findall(text.lower())


def remove_stopwords(tokens: list) -> list:
    """
    Remove common English stopwords from the list of tokens.
    
    Args:
        tokens: List of tokens to filter
        
    Returns:
        List of tokens with stopwords removed
    """
    return [token for token in tokens if token not in STOPWORDS]


def stem_tokens(tokens: list) -> list:
    """
    Apply Porter stemming to each token.
    
    Args:
        tokens: List of tokens to stem
        
    Returns:
        List of stemmed tokens
    """
    return [ps.stem(token) for token in tokens]


def preprocess(text: str) -> list:
    """
    Apply the full preprocessing pipeline: tokenization, stopword removal, and stemming.
    
    Args:
        text: Input text to preprocess
        
    Returns:
        List of preprocessed tokens (lowercase, no stopwords, stemmed)
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    return stem_tokens(tokens)
