import os
import json
import pickle
import logging
import subprocess
from typing import List
from data.document import Document

# File paths for caching and storing data
CACHE_FILE = os.path.join(os.path.dirname(__file__), "arxiv_cache.pkl")
METADATA_FILE = os.path.join(os.path.dirname(__file__), "arxiv-metadata-oai-snapshot.json")

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def download_metadata_file() -> None:
    """
    Download ArXiv metadata from Kaggle with progress feedback.
    
    This function attempts to download using the Kaggle CLI first,
    then falls back to curl if Kaggle CLI is not available.
    """
    logger.info("Metadata file not found. Downloading from Kaggle...")
    
    # Try Kaggle CLI first
    kaggle_command = [
        "kaggle", "datasets", "download", "-d", "Cornell-University/arxiv",
        "-p", os.path.dirname(METADATA_FILE), "--unzip"
    ]
    
    try:
        logger.info("Initiating download via Kaggle CLI...")
        # Don't capture output so CLI progress is visible to user
        result = subprocess.run(kaggle_command)
        if result.returncode != 0:
            logger.error("Kaggle CLI returned an error, attempting fallback...")
            raise Exception("Dataset download via Kaggle CLI failed.")
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
        
        logger.info("Download via curl complete. Extracting zip file...")
        unzip_command = ["unzip", "-o", zip_path, "-d", os.path.dirname(METADATA_FILE)]
        result = subprocess.run(unzip_command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Unzip failed. Output:")
            logger.error(result.stderr)
            raise Exception("Unzip of downloaded dataset failed.")
    
    logger.info("Download complete.")


def load_arxiv_dataset(sample_size: int = 1000, use_cache: bool = True) -> List[Document]:
    """
    Load the ArXiv dataset from a JSON file or cache.
    
    This function handles caching, loading from disk, or downloading from Kaggle.
    
    Args:
        sample_size: Maximum number of documents to load
        use_cache: Whether to use cached data if available
    
    Returns:
        List of Document objects
    
    Raises:
        FileNotFoundError: If metadata file can't be found or downloaded
    """
    # Try to load from cache first
    if use_cache and os.path.exists(CACHE_FILE):
        logger.info("Loading documents from cache...")
        with open(CACHE_FILE, "rb") as f:
            documents = pickle.load(f)
        return documents[:sample_size]

    # If no cache, try to load from file or download
    if not os.path.exists(METADATA_FILE):
        download_metadata_file()
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file still not found at {METADATA_FILE} after download.")

    logger.info("Loading dataset from ArXiv metadata file...")
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
    
    logger.info(f"Loaded {len(documents)} documents from ArXiv metadata.")
    
    # Cache for future use
    if use_cache:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(documents, f)
        logger.info(f"Cached documents to {CACHE_FILE}.")
    
    return documents


if __name__ == "__main__":
    # Quick test
    docs = load_arxiv_dataset(sample_size=10)
    print(f"Processed {len(docs)} documents:")
    for doc in docs[:3]:
        print(f"- {doc.doc_id}: {doc.title[:50]}...")