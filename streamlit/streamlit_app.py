"""
Streamlit App for ArQiv Search Engine.

This module initializes and runs the Streamlit interface for the ArQiv search engine.
"""

import sys
import os
import time
import streamlit as st
import pandas as pd
import plotly.express as px

# Ensure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_arxiv_dataset
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25Ranker
from ranking.tfidf import TFIDFRanker
from ranking.fast_vector_ranker import FastVectorRanker
from search.fuzzy_search import fuzzy_search

# Page configuration and custom CSS injection
st.set_page_config(page_title="ArQiv Search", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
      /* Overall app styling */
      .main { background-color: #f9f9f9; padding: 20px; }
      /* Header banner */
      .header-banner {
          background: linear-gradient(90deg, #2e8b57, #6abf69);
          color: white;
          padding: 25px;
          text-align: center;
          border-radius: 8px;
          margin-bottom: 20px;
      }
      /* Sidebar info box */
      .sidebar-info {
          font-size: 14px;
          color: #555;
      }
      /* Result panel styling */
      .result-panel {
          background-color: #fff;
          padding: 15px;
          border: 1px solid #ddd;
          border-radius: 8px;
          margin-bottom: 10px;
      }
      .result-panel p { margin: 4px 0; }
      details { cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for URL parameters
if "url_params_initialized" not in st.session_state:
    # Get query params from URL
    if "query" in st.query_params:
        st.session_state["search_query"] = st.query_params["query"]
    else:
        st.session_state["search_query"] = ""
        
    if "mode" in st.query_params:
        st.session_state["search_mode"] = st.query_params["mode"]
    else:
        st.session_state["search_mode"] = "BM25"
        
    st.session_state["url_params_initialized"] = True

@st.cache_resource(show_spinner=True)
def init_search_engine(sample_size=1000):
    """Initialize and cache search engine components."""
    docs = load_arxiv_dataset(sample_size=sample_size)
    index = InvertedIndex()
    if not index.load_from_file():
        for doc in docs:
            index.index_document(doc)
        index.save_to_file()
    bm25_ranker = BM25Ranker(docs, index)
    tfidf_ranker = TFIDFRanker(docs)
    fast_vector_ranker = FastVectorRanker(docs)
    return docs, index, bm25_ranker, tfidf_ranker, fast_vector_ranker


docs, index, bm25_ranker, tfidf_ranker, fast_vector_ranker = init_search_engine()

# Display header banner
st.markdown(
    "<div class='header-banner'><h1>Welcome to ArQiv Search Engine</h1>"
    "<p>Advanced, Fast & Scalable Academic Search</p></div>",
    unsafe_allow_html=True,
)

# Sidebar with enhanced options and documentation
with st.sidebar:
    st.header("Search Options")
    search_mode = st.radio(
        "Select Search Mode",
        ("Basic Boolean", "BM25", "TF-IDF", "Fast Vector", "Fuzzy"),
        index=["Basic Boolean", "BM25", "TF-IDF", "Fast Vector", "Fuzzy"].index(
            st.session_state.get("search_mode", "BM25")
        )
    )
    
    # Store search mode in session state
    st.session_state["search_mode"] = search_mode
    
    show_autocomplete = st.checkbox("Show Autocomplete Suggestions", value=True)
    
    # Add filter options
    st.markdown("### Filter Results")
    with st.expander("Category Filters"):
        categories = ["cs.AI", "cs.LG", "cs.CL", "math.ST", "physics.comp-ph"]
        selected_categories = st.multiselect(
            "Select categories to include",
            options=categories,
            default=[]
        )
    
    st.markdown("---")
    # Documentation in sidebar
    with st.expander("About Search Modes"):
        st.markdown("""
        - **Basic Boolean**: Simple AND search returning documents containing all query terms
        - **BM25**: Probabilistic relevance ranking (similar to modern search engines)
        - **TF-IDF**: Term frequency-inverse document frequency ranking
        - **Fast Vector**: Quick approximate semantic ranking
        - **Fuzzy**: Handles spelling errors and variations
        """)


def main():
    """Main function to run the search interface."""
    # Main search bar with URL parameter handling
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([5,1])
        with col1:
            search_query = st.text_input(
                "Enter your search query",
                value=st.session_state.get("search_query", ""),
                placeholder="Type your query here...",
                key="query_input",
                help="Enter terms to search for in the ArXiv corpus"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🔍 Search", use_container_width=True):
                # Update URL parameters when search button is clicked
                st.query_params.query = search_query
                st.query_params.mode = search_mode
                
    # Store current query
    st.session_state["search_query"] = search_query

    # Dynamic Autocomplete with improved styling
    if search_query and show_autocomplete:
        suggestions = index.trie.starts_with(search_query.lower())
        if suggestions:
            st.markdown('<div style="background-color:#f5f5f5; padding:10px; border-radius:5px;">', unsafe_allow_html=True)
            st.markdown("#### Autocomplete Suggestions")
            cols = st.columns(5)
            for i, suggestion in enumerate(suggestions[:10]):
                if cols[i % 5].button(suggestion, key=f"sugg_{suggestion}"):
                    st.session_state["search_query"] = suggestion
                    # Update URL parameters for the suggestion
                    st.query_params.query = suggestion
                    st.query_params.mode = search_mode
                    st.markdown(
                        "<script>window.location.reload()</script>",
                        unsafe_allow_html=True,
                    )
                    st.stop()
            st.markdown('</div>', unsafe_allow_html=True)

    if search_query:
        with st.spinner("Searching..."):
            start_time = time.perf_counter()
            results = None
            if search_mode == "Basic Boolean":
                results = sorted(index.search(search_query))
            elif search_mode == "BM25":
                scores = bm25_ranker.rank(search_query)
                filtered_scores = {doc_id: score for doc_id, score in scores.items() if score > 0}
                results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
            elif search_mode == "TF-IDF":
                scores = tfidf_ranker.rank(search_query)
                filtered_scores = {doc_id: score for doc_id, score in scores.items() if score > 0}
                results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
            elif search_mode == "Fast Vector":
                ranking = fast_vector_ranker.rank(search_query, top_k=len(docs))
                filtered_scores = {doc_id: score for doc_id, score in ranking.items() if score > 0}
                results = sorted(filtered_scores.items(), key=lambda x: x[1], reverse=True)
            elif search_mode == "Fuzzy":
                results = sorted(fuzzy_search(search_query, index))
            elapsed = time.perf_counter() - start_time

        # Enhanced tabs with icons
        tabs = st.tabs(["📝 Results", "📊 Visualization", "📈 Analytics"])

        with tabs[0]:
            st.markdown(f"**Search completed in {elapsed:.4f} seconds.**")
            st.markdown("### Search Results")
            if results:
                import re
                query_tokens = [re.escape(token) for token in search_query.split() if token.strip()]
                pattern = re.compile("|".join(query_tokens), re.IGNORECASE) if query_tokens else None
                for res in (results if isinstance(results, list) else list(results.items())):
                    if isinstance(res, tuple):
                        doc_id, score = res
                    else:
                        doc_id, score = res, None
                    doc = next((d for d in docs if d.doc_id == doc_id), None)
                    if doc:
                        full_abstract = doc.content if doc.content else "No abstract available."
                        if pattern:
                            highlighted_abstract = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", full_abstract)
                        else:
                            highlighted_abstract = full_abstract
                        st.markdown(
                            f"""
                            <div style="padding:10px 0; border-bottom:1px solid #ccc;">
                                <p style="margin:2px 0; font-size:14px;"><strong>ID:</strong> {doc.doc_id}</p>
                                <p style="margin:2px 0; font-size:16px;"><strong>Title:</strong> {doc.title}</p>
                                {"<p style='margin:2px 0; font-size:14px;'><strong>Score:</strong> " + f"{score:.4f}" + "</p>" if score is not None else ""}
                                <p style="margin:2px 0; font-size:14px;"><strong>Authors:</strong> {doc.metadata.get('authors', 'N/A')}</p>
                                <p style="margin:2px 0; font-size:14px;"><strong>Categories:</strong> {doc.metadata.get('categories', 'N/A')}</p>
                                <p style="margin:2px 0; font-size:14px;">
                                    <strong>Link:</strong> <a href="https://arxiv.org/abs/{doc.doc_id}" target="_blank" style="color: #2e8b57;">https://arxiv.org/abs/{doc.doc_id}</a>
                                </p>
                                <p style="margin:2px 0; font-size:14px;"><strong>Abstract:</strong></p>
                                <p style="margin:2px 0; font-size:14px;">{highlighted_abstract}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown("**No results found.**")

            # Apply category filter if selected
            if selected_categories and results:
                filtered_results = []
                for res in (results if isinstance(results, list) else list(results.items())):
                    if isinstance(res, tuple):
                        doc_id, score = res
                    else:
                        doc_id, score = res, None
                        
                    doc = next((d for d in docs if d.doc_id == doc_id), None)
                    if doc:
                        doc_categories = doc.metadata.get('categories', '')
                        if any(cat in doc_categories for cat in selected_categories):
                            filtered_results.append(res)
                
                results = filtered_results

        with tabs[1]:
            st.markdown("### Visualization")
            if search_mode in ["BM25", "TF-IDF", "Fast Vector"] and results and isinstance(results, list):
                # Enhanced visualization with Plotly
                df = pd.DataFrame(results, columns=["doc_id", "score"])
                
                # Plotly bar chart for better interactivity
                fig = px.bar(
                    df.sort_values("score", ascending=False).head(15), 
                    x="doc_id", 
                    y="score",
                    title=f"Top 15 Document Scores for Query: '{search_query}'",
                    labels={"doc_id": "Document ID", "score": "Relevance Score"},
                    color="score",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show score distribution
                st.markdown("#### Score Distribution")
                hist_fig = px.histogram(
                    df, 
                    x="score", 
                    nbins=20,
                    title="Distribution of Relevance Scores"
                )
                st.plotly_chart(hist_fig, use_container_width=True)
            else:
                st.markdown("Visualization not available for this mode or no results.")
                
        with tabs[2]:
            st.markdown("### Search Analytics")
            if results:
                # Calculate analytics on results
                result_count = len(results)
                avg_score = 0
                
                if search_mode in ["BM25", "TF-IDF", "Fast Vector"] and isinstance(results, list):
                    if results and isinstance(results[0], tuple):
                        scores = [score for _, score in results]
                        avg_score = sum(scores) / len(scores) if scores else 0
                
                # Display stats in metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Documents Found", result_count)
                col2.metric("Average Score", f"{avg_score:.4f}")
                col3.metric("Search Time", f"{elapsed:.4f} sec")
                
                # Show query term frequency
                st.markdown("#### Query Term Analysis")
                if query_tokens:
                    term_counts = {}
                    for doc_id in (doc_id for doc_id, _ in results) if isinstance(results[0], tuple) else results:
                        doc = next((d for d in docs if d.doc_id == doc_id), None)
                        if doc:
                            for token in query_tokens:
                                if token in term_counts:
                                    term_counts[token] += 1
                                else:
                                    term_counts[token] = 1
                    
                    term_df = pd.DataFrame({
                        "Term": list(term_counts.keys()),
                        "Frequency": list(term_counts.values())
                    })
                    
                    term_fig = px.bar(
                        term_df.sort_values("Frequency", ascending=False),
                        x="Term",
                        y="Frequency",
                        title="Query Term Frequency in Results"
                    )
                    st.plotly_chart(term_fig, use_container_width=True)


if __name__ == "__main__":
    main()