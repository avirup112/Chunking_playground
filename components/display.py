import streamlit as st
from typing import List, Dict, Any, Optional
from utils.visualization import (
    display_chunk_statistics,
    display_chunks_with_highlighting,
    display_retrieval_results,
    compare_retrieval_strategies
)

def display_document_info(text: str, filename: str):
    """Display information about the uploaded document."""
    st.subheader("📄 Document Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Filename", filename)
    
    with col2:
        st.metric("Characters", len(text))
    
    with col3:
        st.metric("Words", len(text.split()))
    
    # Show document preview
    with st.expander("Document Preview"):
        st.text_area(
            "Document Content",
            text[:1000] + "..." if len(text) > 1000 else text,
            height=200,
            disabled=True
        )

def display_chunking_results(chunks: List[Dict[str, Any]], strategy: str):
    """Display chunking results."""
    st.subheader(f"🔪 Chunking Results - {strategy.title()}")
    
    if not chunks:
        st.warning("No chunks generated.")
        return
    
    # Display statistics
    display_chunk_statistics(chunks)
    
    # Display chunks
    display_chunks_with_highlighting(chunks, key_prefix="chunking")

def display_query_interface():
    """Display query input interface."""
    st.subheader("🔍 Query Interface")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What is the main topic of this document?",
        key="query_input"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        search_button = st.button("Search", type="primary")
    with col2:
        llm_button = st.button("Ask LLM", type="secondary", help="Get AI-powered answer using retrieved chunks")
    
    return query, search_button, llm_button

def display_retrieval_interface(results: List[Dict[str, Any]], query: str, retriever_type: str):
    """Display retrieval results interface."""
    if not query:
        st.info("Enter a query above to see retrieval results.")
        return
    
    st.subheader(f"📊 Retrieval Results - {retriever_type.upper()}")
    
    if not results:
        st.warning("No results found for your query.")
        return
    
    display_retrieval_results(results, query, "main")

def display_comparison_interface(strategies_results: Dict[str, List[Dict[str, Any]]], query: str):
    """Display comparison between different strategies."""
    if not query or not strategies_results:
        return
    
    st.subheader("⚖️ Strategy Comparison")
    
    # Create tabs for different comparisons
    tab1, tab2 = st.tabs(["Retrieval Comparison", "Strategy Details"])
    
    with tab1:
        compare_retrieval_strategies(strategies_results)
    
    with tab2:
        for strategy, results in strategies_results.items():
            with st.expander(f"{strategy.upper()} Results"):
                display_retrieval_results(results, query, f"comp_{strategy}")

def display_educational_content(chunks: List[Dict[str, Any]], results: List[Dict[str, Any]], strategy: str, retriever_type: str):
    """Display educational explanations."""
    st.subheader("🎓 Educational Insights")
    
    with st.expander("Why this chunking strategy?"):
        explanations = {
            "fixed_length": "Fixed-length chunking creates uniform chunks but may split sentences awkwardly. Good for consistent processing but may lose context.",
            "overlapping": "Overlapping (sliding window) chunks preserve context at boundaries, reducing information loss but increasing storage requirements.",
            "recursive": "Recursive splitting respects natural text boundaries (paragraphs, sentences) for better semantic coherence.",
            "semantic": "Semantic chunking groups related content together based on meaning, potentially improving retrieval relevance.",
            "sentence": "Sentence-based chunking preserves complete thoughts but may create very small or large chunks.",
            "paragraph": "Paragraph-based chunking maintains natural document structure but chunk sizes can vary significantly.",
            "heading": "Heading-based chunking follows document structure (headings, sections) for logical organization.",
            "topic": "Topic-based chunking uses clustering to group semantically similar content together.",
            "dynamic_ai": "AI-powered chunking uses language models to determine optimal boundaries based on semantic coherence."
        }
        
        st.write(explanations.get(strategy, "Custom chunking strategy."))
    
    with st.expander("Why this retriever?"):
        retriever_explanations = {
            "bm25": "BM25 is a classical keyword-based retriever. Good for exact term matching but may miss semantic similarity.",
            "faiss": "FAISS uses dense vector similarity. Better for semantic matching but requires good embeddings.",
            "chroma": "Chroma provides persistent vector storage with metadata support. Good for production applications.",
            "hybrid": "Hybrid retrieval combines keyword (BM25) and semantic (dense) search for comprehensive coverage.",
            "mmr": "Maximal Marginal Relevance balances relevance and diversity to avoid redundant results.",
            "reranking": "Two-stage retrieval: initial candidates are reranked using more sophisticated similarity measures.",
            "multi_vector": "Multi-vector retrieval uses different representations (original, summary, keywords) for each chunk."
        }
        
        st.write(retriever_explanations.get(retriever_type, "Custom retriever."))
    
    if chunks and results:
        chunk_count = len(chunks)
        result_count = len(results)
        
        st.info(f"""
        **Performance Summary:**
        - Generated {chunk_count} chunks using {strategy} strategy
        - Retrieved {result_count} relevant chunks using {retriever_type}
        - Average chunk size: {sum(len(c['text']) for c in chunks) // len(chunks)} characters
        """)
