import sys
import time
import os
from collections import OrderedDict
import concurrent.futures

from rich.console import Console
from rich.prompt import Prompt, IntPrompt
from rich.progress import track, Progress
from rich.text import Text
from rich.highlighter import Highlighter
from rich.panel import Panel
from rich.markdown import Markdown

from data.preprocessing import preprocess, tokenize
from data.loader import load_arxiv_dataset
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25Ranker
from ranking.tfidf import TFIDFRanker
from search.fuzzy_search import fuzzy_search
from ranking.fast_vector_ranker import FastVectorRanker
# Import BERTRanker only when needed to avoid startup delays
# from ranking.bert_ranker import BERTRanker

# Configure console for rich output
console = Console()

# Constants
QUERY_CACHE_CAPACITY = 100
TOP_K_RESULTS = 10


def display_banner() -> None:
    """Display the ArQiv banner and welcome message."""
    banner = r"""
   _            ____  _        
  /_\   _ __   /___ \(_)__   __
 //_\\ | '__| //  / /| |\ \ / /
/  _  \| |   / \_/ / | | \ V / 
\_/ \_/|_|   \___,_\ |_|  \_/  
    """
    instructions = (
        "Welcome to ArQiv - The Ultimate ArXiv Search Engine!\n"
        "Explore cutting-edge research with lightning-fast search functionality."
    )
    banner_panel = Panel(banner + "\n" + instructions, border_style="bright_green")
    console.print(banner_panel)


class QueryCache:
    """
    LRU cache for query results.
    
    Stores the most recent query results for instant retrieval.
    
    Attributes:
        capacity: Maximum number of queries to cache
        cache: OrderedDict mapping query keys to results
    """
    def __init__(self, capacity: int = QUERY_CACHE_CAPACITY):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: str):
        """Get result for a query key, moving it to most recently used position."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value) -> None:
        """
        Set result for a query key, evicting least recently used if at capacity.
        
        Args:
            key: Query key (format: "algorithm-query")
            value: Query result to cache
        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class QueryHighlighter(Highlighter):
    """
    Highlighter for query terms in document snippets.
    
    This class highlights query terms in snippets to improve readability
    and help users identify relevance.
    
    Attributes:
        query_terms: List of terms to highlight
    """
    def __init__(self, query_terms):
        self.query_terms = query_terms
        super().__init__()
    
    def highlight(self, text: Text) -> Text:
        """
        Apply highlighting to all query terms in the text.
        
        Args:
            text: Rich Text object to highlight
            
        Returns:
            Highlighted text
        """
        plain = text.plain
        for term in self.query_terms:
            start = 0
            lower_plain = plain.lower()
            lower_term = term.lower()
            while True:
                index = lower_plain.find(lower_term, start)
                if index == -1:
                    break
                text.stylize("bold reverse yellow", index, index + len(term))
                start = index + len(term)
        return text


# Initialize global query cache
query_cache = QueryCache()


def load_documents():
    """
    Load documents from the ArXiv dataset.
    
    Returns:
        List of Document objects
    """
    console.print("[bold green]Loading dataset from Kaggle ArXiv metadata...[/bold green]")
    # Adjust sample_size as needed for performance vs. coverage
    docs = load_arxiv_dataset(sample_size=1000)
    # Show download/processing progress feedback.
    for _ in track(range(1), description="Processing documents..."):
        pass
    console.print(f"[bold green]Loaded {len(docs)} documents.[/bold green]")
    return docs


def generate_snippet(doc, query, index, max_length=120):
    """
    Generate a snippet from document content, highlighting query terms.
    
    This mimics Google's search result snippets by showing text
    around matched query terms with highlighting.
    
    Args:
        doc: Document to generate snippet for
        query: Query string
        index: Inverted index for term positions
        max_length: Maximum snippet length in characters
        
    Returns:
        Rich Text object with highlighted query terms
    """
    if not doc or not doc.content:
        return Text("")
    
    query_tokens = preprocess(query)
    if not query_tokens:
        return Text(doc.content[:max_length])
    
    # Get positions where query terms appear in the document
    positions = index.get_term_highlights(doc.doc_id, query)
    if not positions:
        return Text(doc.content[:max_length])
    
    # Find a suitable window around the matched terms
    tokens = tokenize(doc.content)
    if not tokens:
        return Text(doc.content[:max_length])
    
    # Start snippet at a position where a query term occurs
    best_pos = positions[0]
    window_size = max_length // 10  # Approximate chars per token
    
    # Find start and end positions for the snippet
    start_token = max(0, best_pos - window_size // 2)
    end_token = min(len(tokens), start_token + window_size)
    
    # Create snippet text with ellipsis if needed
    content_tokens = tokenize(doc.content)
    snippet = "... " if start_token > 0 else ""
    snippet += " ".join(content_tokens[start_token:end_token])
    snippet += " ..." if end_token < len(content_tokens) else ""
        
    # Create rich text and highlight query terms
    text = Text(snippet)
    highlighter = QueryHighlighter(query_tokens)
    return highlighter.highlight(text)


def display_results(title, results, docs_by_id=None, search_time=None, query="", index=None):
    """
    Display search results in rich panels with metadata and highlighting.
    
    Args:
        title: Title for the result set
        results: Dictionary mapping doc_ids to scores or set of doc_ids
        docs_by_id: Dictionary mapping doc_ids to Document objects
        search_time: Search time in seconds
        query: Query string for highlighting
        index: Inverted index for term positions
    """
    console.print(f"\n[bold underline]{title}[/bold underline]")
    if search_time is not None:
        console.print(f"[italic green]Search completed in {search_time:.4f} seconds[/italic green]\n")
    
    # Handle ranked results (dictionary with scores)
    if isinstance(results, dict):
        for doc_id, score in results.items():
            doc = docs_by_id.get(doc_id, None)
            if not doc:
                continue
                
            # Generate snippets of different lengths
            abstract_snippet = generate_snippet(doc, query, index, max_length=500).plain if doc.content else "N/A"
            content_snippet = generate_snippet(doc, query, index, max_length=150).plain if doc.content else "N/A"
            
            # Build ArXiv link from document ID
            link = f"https://arxiv.org/abs/{doc_id}"
            
            # Get metadata
            authors = doc.metadata.get("authors", "N/A")
            categories = doc.metadata.get("categories", "N/A")
            
            # Create rich panel with document details
            detail = (
                f"[bold]ID:[/bold] {doc_id}\n"
                f"[bold]Title:[/bold] {doc.title}\n"
                f"[bold]Score:[/bold] {score:.4f}\n"
                f"[bold]Authors:[/bold] {authors}\n"
                f"[bold]Categories:[/bold] {categories}\n"
                f"[bold]Link:[/bold] [underline blue]{link}[/underline blue]\n"
                f"[bold]Abstract:[/bold] {abstract_snippet}\n"
                f"[bold]Content Snippet:[/bold] {content_snippet}"
            )
            console.print(Panel(detail, border_style="magenta"))
    # Handle unranked results (set or list of doc_ids)
    else:
        for doc_id in results:
            doc = docs_by_id.get(doc_id, None)
            if not doc:
                continue
                
            # Generate snippets
            abstract_snippet = generate_snippet(doc, query, index, max_length=500).plain if doc.content else "N/A"
            content_snippet = generate_snippet(doc, query, index, max_length=150).plain if doc.content else "N/A"
            
            link = f"https://arxiv.org/abs/{doc_id}"
            authors = doc.metadata.get("authors", "N/A")
            categories = doc.metadata.get("categories", "N/A")
            
            # Create panel without score
            detail = (
                f"[bold]ID:[/bold] {doc_id}\n"
                f"[bold]Title:[/bold] {doc.title}\n"
                f"[bold]Authors:[/bold] {authors}\n"
                f"[bold]Categories:[/bold] {categories}\n"
                f"[bold]Link:[/bold] [underline blue]{link}[/underline blue]\n"
                f"[bold]Abstract:[/bold] {abstract_snippet}\n"
                f"[bold]Content Snippet:[/bold] {content_snippet}"
            )
            console.print(Panel(detail, border_style="magenta"))


def main():
    """Main function to run the ArQiv search engine."""
    # Load documents and build index
    start_load = time.time()
    docs = load_documents()
    load_time = time.time() - start_load
    console.print(f"[bold blue]Documents loaded in {load_time:.2f} seconds.[/bold blue]")
    
    # Create document lookup dictionary
    docs_by_id = {doc.doc_id: doc for doc in docs}
    
    # Initialize inverted index
    index = InvertedIndex()
    console.print("[bold]Initializing search engine... This is a one-time process.[/bold]")
    start_index = time.time()
    
    # Attempt to load index from disk, rebuild if not found
    if not index.load_from_file():
        for doc in track(docs, description="Building optimized index..."):
            index.index_document(doc)
        index.save_to_file()
    
    index_time = time.time() - start_index
    console.print(f"[bold green]Search engine ready with {len(index.index)} terms in {index_time:.2f} seconds.[/bold green]")
    
    # Initialize lightweight rankers
    console.print("[bold yellow]Initializing BM25, TF-IDF and Fast Vector Rankers... (one time cost)[/bold yellow]")
    bm25_ranker = BM25Ranker(docs, index)
    tfidf_ranker = TFIDFRanker(docs)
    fast_vector_ranker = FastVectorRanker(docs)
    bert_ranker = None  # Initialize only if user explicitly requests it
    
    # Show banner after initialization
    display_banner()
    
    # Main interaction loop
    while True:
        # Show menu
        console.print("\n[bold]ArQiv Search Menu:[/bold]")
        console.print("1) Basic Boolean Search [Fast: ~5ms]")
        console.print("2) BM25 Ranking [Accurate: Slow, ~50-100ms]")
        console.print("3) Fuzzy Search [Approximate]")
        console.print("4) TF-IDF Ranking [Very Fast: <5ms]")
        console.print("5) Fast Vector-Ranking [Very Fast: ~<5ms]")
        console.print("6) BERT Ranking [Neural: ~100-200ms] (optional, init takes time)")
        console.print("0) Exit ArQiv")
        
        # Get user choice
        choice = IntPrompt.ask("Your choice", choices=["0", "1", "2", "3", "4", "5", "6"])
        if choice == 0:
            console.print("[bold red]Exiting ArQiv...[/bold red]")
            break
            
        # Get query from user
        query = Prompt.ask("Enter search query")
        
        # Check cache for results
        start_search = time.perf_counter()
        cache_key = f"{choice}-{query}"
        cached_result = query_cache.get(cache_key)
        
        # If cached, show results immediately
        if cached_result is not None:
            end_search = time.perf_counter()
            display_results("Cached Results (Top 10)", cached_result, docs_by_id, end_search - start_search, query, index)
            continue
        
        # Process search with progress indicator
        with Progress() as progress:
            task = progress.add_task("[cyan]Searching...", total=100)
            
            if choice == 1:
                # Boolean search
                progress.update(task, completed=10, description="Preprocessing query...")
                _ = preprocess(query)
                progress.update(task, completed=30, description="Performing boolean search...")
                results_set = index.search(query)
                progress.update(task, completed=80, description="Sorting results...")
                result = sorted(results_set)[:TOP_K_RESULTS]
                progress.update(task, completed=100)
            
            elif choice == 2:
                # BM25 ranking
                progress.update(task, completed=10, description="Ranking with BM25...")
                scores = bm25_ranker.rank(query)
                progress.update(task, completed=90, description="Sorting BM25 results...")
                sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RESULTS])
                result = sorted_scores
                progress.update(task, completed=100)
            
            elif choice == 3:
                # Fuzzy search
                progress.update(task, completed=20, description="Preprocessing query...")
                query_tokens = preprocess(query)
                progress.update(task, completed=40, description="Performing fuzzy search...")
                result = sorted(fuzzy_search(query, index))[:TOP_K_RESULTS]
                progress.update(task, completed=100)
            
            elif choice == 4:
                # TF-IDF ranking
                progress.update(task, completed=10, description="Ranking with TF-IDF...")
                scores = tfidf_ranker.rank(query)
                progress.update(task, completed=90, description="Sorting TF-IDF results...")
                sorted_scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RESULTS])
                result = sorted_scores
                progress.update(task, completed=100)
            
            elif choice == 5:
                # Fast vector ranking
                progress.update(task, completed=10, description="Ranking with Fast Vector-Ranker...")
                ranking = fast_vector_ranker.rank(query, top_k=TOP_K_RESULTS)
                result = ranking
                progress.update(task, completed=100)
            
            elif choice == 6:
                # BERT ranking - optional, initialize on demand
                progress.stop()
                if bert_ranker is None:
                    use_bert = Prompt.ask("BERT ranking is resource heavy on CPU. Initialize BERT Ranker? (y/n)", choices=["y", "n"])
                    if use_bert.lower() == "y":
                        console.print("[bold yellow]Initializing BERT Ranker... This may take a while.[/bold yellow]")
                        # Import here to avoid loading at startup
                        from ranking.bert_ranker import BERTRanker
                        bert_ranker = BERTRanker(docs)
                    else:
                        console.print("[bold red]BERT ranking cancelled.[/bold red]")
                        continue
                        
                # Use separate progress bar to avoid interference with BERT output
                with Progress() as bert_progress:
                    bert_task = bert_progress.add_task("[cyan]Ranking with BERT Ranker...", total=100)
                    bert_progress.update(bert_task, completed=10)
                    ranking = bert_ranker.rank(query, top_k=TOP_K_RESULTS)
                    result = ranking
                    bert_progress.update(bert_task, completed=100)
        
        # Measure search time and cache results
        end_search = time.perf_counter()
        query_cache.set(cache_key, result)
        
        # Display results
        display_results("Search Results (Top 10)", result, docs_by_id, end_search - start_search, query, index)


if __name__ == "__main__":
    main()