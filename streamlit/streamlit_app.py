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

st.set_page_config(page_title="ArQiv Search", layout="wide", initial_sidebar_state="expanded")

# custom CSS injection

st.markdown(
    """
    <style>
      /* Overall app styling */
      .main { background-color: #f9f9f9; padding: 20px; }
      
      /* Header banner */
      .header-banner {
          background: linear-gradient(135deg, #1a5276, #27ae60, #2e86c1);
          background-size: 300% 300%;
          animation: gradient-animation 10s ease infinite;
          color: white;
          padding: 25px;
          text-align: center;
          border-radius: 8px;
          margin-bottom: 20px;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
      }
      .header-banner h1 {
          font-size: 3.2rem;
          font-weight: 700;
          margin-bottom: 0;
          text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
          letter-spacing: 2px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
      }
      /* Custom search text styling for logo */
      .header-banner .q-text {
          position: relative;
          display: inline-block;
          bottom: 4px;
      }
      .header-banner p {
          font-size: 1.2rem;
          margin-top: 8px;
          opacity: 0.9;
          font-weight: 300;
          letter-spacing: 1px;
      }
      
      /* Sidebar info box */
      .sidebar-info {
          font-size: 14px;
          color: #555;
      }
      
      /* Result panel styling - minimalistic */
      .result-panel {
          border-bottom: 1px solid #eee;
          padding: 15px 0;
          margin-bottom: 5px;
          transition: all 0.2s ease;
      }
      .result-panel:hover {
          background-color: #f9f9f9;
      }
      .result-panel p { margin: 4px 0; }
      
      /* Search container styling - minimalistic */
      .search-container {
          margin-bottom: 20px;
          border-bottom: 1px solid #eee;
          padding-bottom: 15px;
      }
      
      /* Search button styling */
      .stButton>button {
          background: #2e8b57;
          color: white;
          border: none;
          padding: 8px 16px;
          border-radius: 4px;
          font-weight: 500;
          transition: all 0.2s ease;
      }
      .stButton>button:hover {
          background: #3a9e6a;
          box-shadow: 0 2px 5px rgba(0,0,0,0.1);
      }
      
      /* Tab styling */
      .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
      }
      .stTabs [data-baseweb="tab"] {
          padding: 8px 12px;
          border-radius: 4px;
      }
      .stTabs [aria-selected="true"] {
          background-color: rgba(46, 139, 87, 0.1);
      }
      
      /* Autocomplete suggestions - minimalistic */
      .autocomplete-container {
          border-left: 3px solid #2e8b57;
          padding: 10px 15px;
          margin-bottom: 15px;
          background-color: #f9f9f9;
      }
      
      /* Metric styling - minimalistic */
      .metric-container {
          padding: 10px 0;
          border-bottom: 1px solid #eee;
      }
      
      /* Details styling */
      details {
          cursor: pointer;
          margin-top: 8px;
      }
      details summary {
          padding: 4px 0;
          display: flex;
          align-items: center;
      }
      details summary:hover {
          color: #2e8b57;
      }
      
      /* Make streamlit elements more minimal */
      div.stMarkdown {
          padding: 0 2px;
      }
      
      /* Material symbol spacing helper */
      .symbol-spacing {
          margin-right: 6px;
      }
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

# Display header banner with emphasis on the "Q" in "ArQiv"
st.markdown(
    '<div class="header-banner"><h1>Ar<span class="q-text"><svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="white"><path d="M200-800v241-1 400-640 200-200Zm0 720q-33 0-56.5-23.5T120-160v-640q0-33 23.5-56.5T200-880h320l240 240v100q-19-8-39-12.5t-41-6.5v-41H480v-200H200v640h241q16 24 36 44.5T521-80H200Zm460-120q42 0 71-29t29-71q0-42-29-71t-71-29q-42 0-71 29t-29 71q0 42 29 71t71 29ZM864-40 756-148q-21 14-45.5 21t-50.5 7q-75 0-127.5-52.5T480-300q0-75 52.5-127.5T660-480q75 0 127.5 52.5T840-300q0 26-7 50.5T812-204L920-96l-56 56Z"/></svg></span>iv</h1>'
    '<p>Advanced, Fast & Scalable Academic Search</p></div>',
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
        st.markdown('<div class="search-container">', unsafe_allow_html=True)
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
            if st.button("Search", icon=":material/search:", use_container_width=True, help="Click to search"):
                # Update URL parameters when search button is clicked
                st.query_params.query = search_query
                st.query_params.mode = search_mode
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display current search mode
        st.markdown(f"<p style='margin-top: -10px; color: #555; font-size: 14px;'>Active mode: <strong>{search_mode}</strong></p>", unsafe_allow_html=True)
                
    # Store current query
    st.session_state["search_query"] = search_query

    # Dynamic Autocomplete with improved styling
    if search_query and show_autocomplete:
        suggestions = index.trie.starts_with(search_query.lower())
        if suggestions:
            st.markdown("#### Autocomplete Suggestions")
            cols = st.columns(5)
            for i, suggestion in enumerate(suggestions[:10]):
                if cols[i % 5].button(suggestion, key=f"sugg_{suggestion}", help=f"Use suggestion: {suggestion}", use_container_width=True):
                    st.session_state["search_query"] = suggestion
                    # Update URL parameters for the suggestion
                    st.query_params.query = suggestion
                    st.query_params.mode = search_mode
                    st.markdown(
                        "<script>window.location.reload()</script>",
                        unsafe_allow_html=True,
                    )
                    st.stop()

    if search_query:
        with st.spinner("Searching..."):
            start_time = time.perf_counter()
            results = None
            if search_mode == "Basic Boolean":
                raw_results = index.search(search_query)
                # Convert to list of doc_ids without scores for consistent handling
                results = list(raw_results) if raw_results else []
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
                fuzzy_results = fuzzy_search(search_query, index)
                # Convert to list of doc_ids without scores for consistent handling
                results = list(fuzzy_results) if fuzzy_results else []
            elapsed = time.perf_counter() - start_time

        # Use Streamlit's native Material Symbols for tabs - without emojis
        tabs = st.tabs([
            "Results", 
            "Visualization", 
            "Analytics"
        ])

        with tabs[0]:
            st.markdown(f"**Search completed in {elapsed:.4f} seconds.**")
            st.header("Search Results")
            if results:
                import re
                query_tokens = [re.escape(token) for token in search_query.split() if token.strip()]
                pattern = re.compile("|".join(query_tokens), re.IGNORECASE) if query_tokens else None
                
                # Handle different result formats properly
                processed_results = []
                
                if isinstance(results, list):
                    if results and isinstance(results[0], tuple):  # Results with scores (BM25, TF-IDF, Vector)
                        processed_results = results
                    else:  # Basic Boolean or Fuzzy (just doc IDs)
                        processed_results = [(doc_id, None) for doc_id in results]
                else:
                    # Handle unexpected result format
                    st.error("Unexpected result format. Please try a different search mode.")
                    
                for doc_id, score in processed_results:
                    doc = next((d for d in docs if d.doc_id == doc_id), None)
                    if doc:
                        full_abstract = doc.content if doc.content else "No abstract available."
                        if pattern:
                            highlighted_abstract = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", full_abstract)
                        else:
                            highlighted_abstract = full_abstract
                            
                        # Add score display for modes that provide scores
                        score_display = f" | Score: {score:.4f}" if score is not None else ""
                            
                        st.markdown(
                            f"""
                            <div class="result-panel">
                                <p style="margin:2px 0; font-size:18px; color:#1a5276;"><strong>{doc.title}</strong></p>
                                <p style="margin:2px 0; font-size:14px; color:#555;">
                                    Authors: {doc.metadata.get('authors', 'N/A')} | 
                                    Categories: {doc.metadata.get('categories', 'N/A')} | 
                                    ID: {doc.doc_id}{score_display}
                                </p>
                                <p style="margin:2px 0; font-size:14px;">
                                    <a href="https://arxiv.org/abs/{doc.doc_id}" target="_blank" style="color: #2e8b57;">
                                        https://arxiv.org/abs/{doc.doc_id}
                                    </a>
                                </p>
                                <details>
                                    <summary style="margin:6px 0; font-size:15px;">Abstract</summary>
                                    <p style="margin:8px 0; font-size:14px; line-height:1.5; color:#333;">{highlighted_abstract}</p>
                                </details>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                st.warning("No results found.")

            # Apply category filter if selected
            if selected_categories and results:
                filtered_results = []
                
                if isinstance(results, list):
                    if results and isinstance(results[0], tuple):  # Results with scores
                        for doc_id, score in results:
                            doc = next((d for d in docs if d.doc_id == doc_id), None)
                            if doc and any(cat in doc.metadata.get('categories', '') for cat in selected_categories):
                                filtered_results.append((doc_id, score))
                    else:  # Basic Boolean or Fuzzy (just doc IDs)
                        for doc_id in results:
                            doc = next((d for d in docs if d.doc_id == doc_id), None)
                            if doc and any(cat in doc.metadata.get('categories', '') for cat in selected_categories):
                                filtered_results.append(doc_id)
                
                results = filtered_results

        with tabs[1]:
            st.header("Visualization")
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
            st.header("Search Analytics")
            if results:
                # Calculate analytics on results
                result_count = len(results)
                avg_score = 0
                
                if search_mode in ["BM25", "TF-IDF", "Fast Vector"] and isinstance(results, list):
                    if results and isinstance(results[0], tuple):
                        scores = [score for _, score in results]
                        avg_score = sum(scores) / len(scores) if scores else 0
                
                # Display stats in metrics without emojis
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Documents", result_count)
                    st.markdown("</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Avg Score", f"{avg_score:.4f}")
                    st.markdown("</div>", unsafe_allow_html=True)
                with col3:
                    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
                    st.metric("Search Time", f"{elapsed:.4f}s")
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Show query term frequency
                st.subheader("Query Term Analysis")
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