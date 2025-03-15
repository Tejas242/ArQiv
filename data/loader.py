import os
import json
import pickle
import logging
import subprocess
from typing import List
from data.document import Document

# File paths for caching and storing data - modified to use central cache directory
def get_cache_file_path(sample_size: int) -> str:
    """Get sample-size specific cache file path in the cache directory."""
    # Use the same cache directory as the inverted index
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"arxiv_cache_{sample_size}.pkl")

METADATA_FILE = os.path.join(os.path.dirname(__file__), "arxiv-metadata-oai-snapshot.json")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def download_metadata_file() -> None:
    """
    Download ArXiv metadata from Kaggle with progress feedback.
    
    Uses Python's built-in zipfile module instead of external unzip command.
    """
    logger.info("Metadata file not found. Downloading from Kaggle...")
    
    # Try Kaggle CLI first
    kaggle_command = [
        "kaggle", "datasets", "download", "-d", "Cornell-University/arxiv",
        "-p", os.path.dirname(METADATA_FILE)
    ]
    
    try:
        logger.info("Initiating download via Kaggle CLI...")
        # Don't capture output so CLI progress is visible to user
        result = subprocess.run(kaggle_command)
        if result.returncode != 0:
            logger.error("Kaggle CLI returned an error, attempting fallback...")
            raise Exception("Dataset download via Kaggle CLI failed.")
            
        # Use Python's zipfile instead of unzip command
        import zipfile
        zip_path = os.path.join(os.path.dirname(METADATA_FILE), "arxiv.zip")
        if os.path.exists(zip_path):
            logger.info("Extracting zip file using Python's zipfile module...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(METADATA_FILE))
            logger.info("Extraction complete.")
        
    except FileNotFoundError:
        # Fall back to curl if Kaggle CLI is not installed
        logger.warning("Kaggle CLI not found. Falling back to curl download...")
        zip_path = os.path.join(os.path.dirname(METADATA_FILE), "arxiv.zip")
        
        # Use curl's progress-bar option for better user feedback
        curl_command = [
            "curl", "-L", "--progress-bar", "-o", zip_path,
            "https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv"
        ]
        
        logger.info("Initiating download via curl...")
        result = subprocess.run(curl_command)
        if result.returncode != 0:
            logger.error("Curl download failed.")
            raise Exception("Dataset download via curl failed.")
        
        # Use Python's zipfile instead of unzip command
        import zipfile
        logger.info("Extracting zip file using Python's zipfile module...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(METADATA_FILE))
        logger.info("Extraction complete.")
    
    logger.info("Download complete.")


def load_arxiv_dataset(sample_size: int = 1000, use_cache: bool = True, verbose: bool = False) -> List[Document]:
    """
    Load the ArXiv dataset from a JSON file or cache.
    
    This function handles caching, loading from disk, or downloading from Kaggle.
    
    Args:
        sample_size: Maximum number of documents to load
        use_cache: Whether to use cached data if available
        verbose: Whether to show detailed log messages
    
    Returns:
        List of Document objects
    
    Raises:
        FileNotFoundError: If metadata file can't be found or downloaded
    """
    # Temporarily reduce logging if not verbose
    if not verbose:
        original_level = logger.level
        logger.setLevel(logging.WARNING)
    else:
        logger.info(f"Loading ArXiv dataset with sample_size={sample_size}")
    
    # Use sample-size specific cache path
    cache_file = get_cache_file_path(sample_size)
    
    # Try to load from cache first
    if use_cache and os.path.exists(cache_file):
        if verbose:
            logger.info(f"Found cache file at {cache_file}, attempting to load...")
        try:
            with open(cache_file, "rb") as f:
                documents = pickle.load(f)
            # If cache has correct size, return it
            if len(documents) == sample_size:
                if verbose:
                    logger.info(f"Successfully loaded {len(documents)} documents from cache")
                # Restore logging level
                if not verbose:
                    logger.setLevel(original_level)
                return documents
            else:
                if verbose:
                    logger.info(f"Cache size mismatch: expected {sample_size}, got {len(documents)}")
        except Exception as e:
            if verbose:
                logger.warning(f"Error loading cache: {str(e)}")

    # If no cache or cache error, try to load from file or download
    if not os.path.exists(METADATA_FILE):
        # Always show download messages
        logger.setLevel(logging.INFO)
        download_metadata_file()
        if not verbose:
            logger.setLevel(logging.WARNING)
            
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file still not found at {METADATA_FILE} after download.")

    if verbose:
        logger.info(f"Loading {sample_size} documents from ArXiv metadata file...")
    documents = []
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            item = json.loads(line)
            doc_id = item.get("id", f"doc_{i}")
            title = item.get("title", "").strip() or "No Title"
            content = item.get("abstract", "").strip()
            # Store all other data as metadata
            metadata = {k: v for k, v in item.items() if k not in {"id", "title", "abstract"}}
            documents.append(Document(doc_id=doc_id, title=title, content=content, metadata=metadata))
    
    if verbose:
        logger.info(f"Loaded {len(documents)} documents from ArXiv metadata file")
    
    # Cache for future use
    if use_cache:
        if verbose:
            logger.info(f"Saving documents to cache at {cache_file}")
        # Ensure directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(documents, f)
        if verbose:
            logger.info(f"Cached {len(documents)} documents to {cache_file}")
    
    # Restore original logging level
    if not verbose:
        logger.setLevel(original_level)
        
    return documents


if __name__ == "__main__":
    # Quick test
    docs = load_arxiv_dataset(sample_size=10)
    print(f"Processed {len(docs)} documents:")
    for doc in docs[:3]:
        print(f"- {doc.doc_id}: {doc.title[:50]}...")