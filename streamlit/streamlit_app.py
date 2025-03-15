"""
Streamlit App for ArQiv Search Engine.

This module initializes and runs the Streamlit interface for the ArQiv search engine.
"""

import sys
import os
import time
import pickle
import logging
import streamlit as st
import pandas as pd
import plotly.express as px

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.loader import load_arxiv_dataset
from index.inverted_index import InvertedIndex
from ranking.bm25 import BM25Ranker
from ranking.tfidf import TFIDFRanker
from ranking.fast_vector_ranker import FastVectorRanker
from search.fuzzy_search import fuzzy_search

st.set_page_config(page_title="ArQiv Search", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .main { background-color: #f9f9f9; padding: 20px; }
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
      @keyframes gradient-animation {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
      }
      .stProgress > div > div > div > div {
          background-color: #2e8b57 !important;
          background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.2) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.2) 50%, rgba(255, 255, 255, 0.2) 75%, transparent 75%, transparent) !important;
          background-size: 20px 20px !important;
          animation: progress-bar-stripes 1s linear infinite !important;
      }
      @keyframes progress-bar-stripes { from { background-position: 40px 0; } to { background-position: 0 0; } }
      .status-box { padding: 10px 15px; border-radius: 5px; background-color: #f8f9fa; border-left: 4px solid #2e8b57; margin-bottom: 16px; animation: fade-in 0.5s ease-in-out; }
      @keyframes fade-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
      .header-banner h1 { font-size: 3.2rem; font-weight: 700; margin-bottom: 0; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); letter-spacing: 2px; display: inline-flex; align-items: center; justify-content: center; }
      .header-banner .q-text { position: relative; display: inline-block; bottom: 4px; }
      .header-banner p { font-size: 1.2rem; margin-top: 8px; opacity: 0.9; font-weight: 300; letter-spacing: 1px; }
      .sidebar-info { font-size: 14px; color: #555; }
      .result-panel { border-bottom: 1px solid #eee; padding: 15px 0; margin-bottom: 5px; transition: all 0.2s ease; }
      .result-panel:hover { background-color: #f9f9f9; }
      .search-container { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px; }
      .stButton>button { background: #2e8b57; color: white; border: none; padding: 8px 16px; border-radius: 4px; font-weight: 500; transition: all 0.2s ease; }
      .stButton>button:hover { background: #3a9e6a; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { padding: 8px 12px; border-radius: 4px; }
      .stTabs [aria-selected="true"] { background-color: rgba(46, 139, 87, 0.1); }
      .autocomplete-container { border-left: 3px solid #2e8b57; padding: 10px 15px; margin-bottom: 15px; background-color: #f9f9f9; }
      .metric-container { padding: 10px 0; border-bottom: 1px solid #eee; }
      details { cursor: pointer; margin-top: 8px; }
      details summary { padding: 4px 0; display: flex; align-items: center; }
      details summary:hover { color: #2e8b57; }
      div.stMarkdown { padding: 0 2px; }
      .symbol-spacing { margin-right: 6px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for URL parameters and search engine components
if "url_params_initialized" not in st.session_state:
    if "query" in st.query_params:
        st.session_state["search_query"] = st.query_params["query"]
    else:
        st.session_state["search_query"] = ""
        
    if "mode" in st.query_params:
        st.session_state["search_mode"] = st.query_params["mode"]
    else:
        st.session_state["search_mode"] = "BM25"
        
    if "sample_size" not in st.session_state:
        st.session_state["sample_size"] = 1000
    
    st.session_state["engine_initialized"] = False
    st.session_state["first_run"] = True
    st.session_state["docs"] = None
    st.session_state["index"] = None
    st.session_state["bm25_ranker"] = None
    st.session_state["tfidf_ranker"] = None
    st.session_state["fast_vector_ranker"] = None
    
    st.session_state["url_params_initialized"] = True
    
    st.session_state["previous_search_mode"] = st.session_state.get("search_mode", "BM25")

def init_search_engine(sample_size=None, force_reload=False):
    sample_size = sample_size or st.session_state.get("sample_size", 1000)
    first_run = st.session_state.get("first_run", True)
    verbose = first_run
    
    if (st.session_state.get("engine_initialized") and 
        st.session_state.get("current_sample_size") == sample_size and 
        not force_reload and not first_run):
        return (
            st.session_state["docs"],
            st.session_state["index"],
            st.session_state["bm25_ranker"],
            st.session_state["tfidf_ranker"],
            st.session_state["fast_vector_ranker"]
        )
    
    loading_container = st.empty()
    
    with loading_container.container():
        if first_run:
            st.markdown('<div class="status-box">', unsafe_allow_html=True)
            st.markdown(f"### Initializing ArQiv Search Engine")
            st.markdown(f"Loading {sample_size} documents and building search indices. This may take a moment...")
            st.markdown('</div>', unsafe_allow_html=True)
            
            cols = st.columns(10)
            for i in range(10):
                with cols[i]:
                    st.markdown(f"<div style='height: 5px; background-color: {'#2e8b57' if i < 5 else '#92dec0'}; margin: 2px; animation: pulse 1.5s infinite {i*0.1}s;'></div>", unsafe_allow_html=True)
        
        with st.spinner(f"Loading search engine components..."):
            original_level = logging.getLogger().level
            if not verbose:
                logging.getLogger().setLevel(logging.WARNING)
            
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            custom_cache_path = os.path.join(cache_dir, f"inverted_index_cache_{sample_size}.pkl")
            
            with st.status("Loading ArXiv dataset", expanded=first_run) as data_status:
                docs = load_arxiv_dataset(sample_size=sample_size, verbose=verbose)
                data_status.update(label="✅ Dataset loaded successfully", state="complete")
            
            index = InvertedIndex()
            
            index_loaded = index.load_from_file(filepath=custom_cache_path, sample_size=sample_size, verbose=verbose)
            
            if not index_loaded:
                if not verbose:
                    logging.getLogger().setLevel(original_level)
                    
                with st.status("Building search index...", expanded=True) as index_status:
                    status_text = "Building search index for the first time..." if first_run else f"Building new index for {sample_size} documents..."
                    st.markdown(f"<p>{status_text}</p>", unsafe_allow_html=True)
                    
                    progress_bar = st.progress(0)
                    total_docs = len(docs)
                    
                    for i, doc in enumerate(docs):
                        index.index_document(doc)
                        if i % max(1, total_docs // 100) == 0:
                            percent = min(1.0, i / total_docs)
                            progress_bar.progress(
                                percent,
                                text=f"Indexing document {i+1}/{total_docs} ({int(percent*100)}%)"
                            )
                    
                    progress_bar.progress(1.0, text="✅ Indexing complete!")
                    
                    st.info("Saving index to disk...")
                    index.save_to_file(filepath=custom_cache_path, sample_size=sample_size)
                    index_status.update(label="✅ Search index built successfully", state="complete")
                
                if not verbose:
                    logging.getLogger().setLevel(logging.WARNING)
            else:
                st.success(f"✅ Loaded existing index with {len(index.index)} terms")
            
            with st.status("Initializing ranking algorithms...", expanded=first_run) as ranker_status:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.spinner("BM25 Ranker"):
                        bm25_ranker = BM25Ranker(docs, index)
                        st.success("BM25 ✓")
                
                with col2:
                    with st.spinner("TF-IDF Ranker"):
                        tfidf_ranker = TFIDFRanker(docs)
                        st.success("TF-IDF ✓")
                
                with col3:
                    with st.spinner("Vector Ranker"):
                        fast_vector_ranker = FastVectorRanker(docs)
                        st.success("Vector ✓")
                        
                ranker_status.update(label="✅ Ranking components initialized", state="complete")
            
            logging.getLogger().setLevel(original_level)
            
            if first_run:
                st.success("🚀 ArQiv search engine initialized successfully! You can now start searching.")
    
    if not first_run:
        loading_container.empty()
    
    st.session_state["first_run"] = False
    
    st.session_state["engine_initialized"] = True
    st.session_state["current_sample_size"] = sample_size
    st.session_state["docs"] = docs
    st.session_state["index"] = index
    st.session_state["bm25_ranker"] = bm25_ranker
    st.session_state["tfidf_ranker"] = tfidf_ranker
    st.session_state["fast_vector_ranker"] = fast_vector_ranker
    
    return docs, index, bm25_ranker, tfidf_ranker, fast_vector_ranker

with st.sidebar:
    st.header("ArQiv Search Engine")
    with st.expander("🔍 Search Configuration", expanded=True):
        st.subheader("Search Mode")
        search_mode = st.radio(
            "Select search algorithm",
            ("Basic Boolean", "BM25", "TF-IDF", "Fast Vector", "Fuzzy"),
            index=["Basic Boolean", "BM25", "TF-IDF", "Fast Vector", "Fuzzy"].index(
                st.session_state.get("search_mode", "BM25")
            ),
            key="search_mode_radio",
            help="Each mode offers different search capabilities."
        )
        if search_mode != st.session_state.get("previous_search_mode"):
            st.session_state["search_mode"] = search_mode
            st.session_state["previous_search_mode"] = search_mode
        st.divider()
        st.subheader("Dataset Size")
        st.info(f"Current: {st.session_state.get('sample_size', 1000)} documents")
        new_sample_size = st.slider(
            "Documents to load", 
            min_value=100, 
            max_value=10000, 
            value=st.session_state.get("sample_size", 1000),
            step=100,
            help="Larger values provide more comprehensive search but require more memory."
        )
        if new_sample_size != st.session_state.get("sample_size", 1000):
            if st.button("Apply Changes", key="apply_sample_size", use_container_width=True):
                with st.spinner("Updating dataset..."):
                    st.session_state["sample_size"] = new_sample_size
                    st.session_state["engine_initialized"] = False
                    st.rerun()
    with st.expander("⚙️ Search Options"):
        show_autocomplete = st.checkbox("Show Autocomplete Suggestions", value=True)
        st.subheader("Result Filters")
        categories = ["cs.AI", "cs.LG", "cs.CL", "math.ST", "physics.comp-ph"]
        selected_categories = st.multiselect("Filter by categories", options=categories, default=[])
    with st.expander("ℹ️ About Search Modes"):
        st.markdown("""
- **Basic Boolean**: AND search returning documents containing all query terms  
- **BM25**: Probabilistic relevance ranking  
- **TF-IDF**: Vector similarity ranking  
- **Fast Vector**: Approximate semantic ranking  
- **Fuzzy**: Handles spelling errors
        """)

progress_placeholder = st.empty()
with progress_placeholder:
    docs, index, bm25_ranker, tfidf_ranker, fast_vector_ranker = init_search_engine()

if not st.session_state.get("first_run", True):
    progress_placeholder.empty()

st.markdown(
    '<div class="header-banner"><h1>Ar<span class="q-text"><svg xmlns="http://www.w3.org/2000/svg" height="48px" viewBox="0 -960 960 960" width="48px" fill="white"><path d="M200-800v241-1 400-640 200-200Zm0 720q-33 0-56.5-23.5T120-160v-640q0-33 23.5-56.5T200-880h320l240 240v100q-19-8-39-12.5t-41-6.5v-41H480v-200H200v640h241q16 24 36 44.5T521-80H200Zm460-120q42 0 71-29t29-71q0-42-29-71t-71-29q-42 0-71 29t-29 71q0 42 29 71t71 29ZM864-40 756-148q-21 14-45.5 21t-50.5 7q-75 0-127.5-52.5T480-300q0-75 52.5-127.5T660-480q75 0 127.5 52.5T840-300q0 26-7 50.5T812-204L920-96l-56 56Z"/></svg></span>iv</h1>'
    '<p>Advanced, Fast & Scalable Academic Search</p></div>',
    unsafe_allow_html=True,
)

def main():
    with st.container():
        dataset_cols = st.columns([3, 2])
        with dataset_cols[0]:
            st.markdown(
                f"""
                <div class="status-box" style="padding: 10px 15px; background-color: #f0f8ff; border-left-color: #1a5276; margin-bottom: 10px;">
                  <strong>{len(docs)}</strong> documents loaded | <strong>{len(index.index)}</strong> terms in index
                </div>
                """,
                unsafe_allow_html=True
            )
        with dataset_cols[1]:
            st.markdown(
                f"""
                <div style="text-align: right; padding-top: 8px; font-size: 14px; color: #666;">
                  Active mode: <strong>{st.session_state.get("search_mode", "BM25")}</strong>
                </div>
                """,
                unsafe_allow_html=True
            )
    
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
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Search", icon=":material/search:", use_container_width=True, help="Click to search"):
                st.query_params.query = search_query
                st.query_params.mode = search_mode
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown(f"<p style='margin-top: -10px; color: #555; font-size: 14px;'>Active mode: <strong>{search_mode}</strong></p>", unsafe_allow_html=True)
                
    st.session_state["search_query"] = search_query

    if search_query and show_autocomplete:
        suggestions = index.trie.starts_with(search_query.lower())
        if suggestions:
            st.markdown("#### Autocomplete Suggestions")
            cols = st.columns(5)
            for i, suggestion in enumerate(suggestions[:10]):
                if cols[i % 5].button(suggestion, key=f"sugg_{suggestion}", help=f"Use suggestion: {suggestion}", use_container_width=True):
                    st.session_state["search_query"] = suggestion
                    st.query_params["query"] = suggestion
                    st.query_params["mode"] = search_mode
                    st.rerun()

    if search_query:
        with st.spinner("Searching..."):
            start_time = time.perf_counter()
            results = None
            if search_mode == "Basic Boolean":
                raw_results = index.search(search_query)
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
                results = list(fuzzy_results) if fuzzy_results else []
            elapsed = time.perf_counter() - start_time

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
                
                processed_results = []
                
                if isinstance(results, list):
                    if results and isinstance(results[0], tuple):
                        processed_results = results
                    else:
                        processed_results = [(doc_id, None) for doc_id in results]
                else:
                    st.error("Unexpected result format. Please try a different search mode.")
                    
                for doc_id, score in processed_results:
                    doc = next((d for d in docs if d.doc_id == doc_id), None)
                    if doc:
                        full_abstract = doc.content if doc.content else "No abstract available."
                        if pattern:
                            highlighted_abstract = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", full_abstract)
                        else:
                            highlighted_abstract = full_abstract
                            
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

            if selected_categories and results:
                filtered_results = []
                
                if isinstance(results, list):
                    if results and isinstance(results[0], tuple):
                        for doc_id, score in results:
                            doc = next((d for d in docs if d.doc_id == doc_id), None)
                            if doc and any(cat in doc.metadata.get('categories', '') for cat in selected_categories):
                                filtered_results.append((doc_id, score))
                    else:
                        for doc_id in results:
                            doc = next((d for d in docs if d.doc_id == doc_id), None)
                            if doc and any(cat in doc.metadata.get('categories', '') for cat in selected_categories):
                                filtered_results.append(doc_id)
                
                results = filtered_results

        with tabs[1]:
            st.header("Visualization")
            if search_mode in ["BM25", "TF-IDF", "Fast Vector"] and results and isinstance(results, list):
                df = pd.DataFrame(results, columns=["doc_id", "score"])
                
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
                result_count = len(results)
                avg_score = 0
                
                if search_mode in ["BM25", "TF-IDF", "Fast Vector"] and isinstance(results, list):
                    if results and isinstance(results[0], tuple):
                        scores = [score for _, score in results]
                        avg_score = sum(scores) / len(scores) if scores else 0
                
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