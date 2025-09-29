import streamlit as st
from config import CHUNKING_STRATEGIES, RETRIEVER_TYPES, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP, DEFAULT_TOP_K

def render_sidebar():
    """Render sidebar with user controls."""
    st.sidebar.header("⚙️ Configuration")
    
    # API Key Section
    st.sidebar.subheader("🔑 API Configuration")
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Enter your OpenAI API key for LLM features (optional)"
    )
    
    # Store API key in session state
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.sidebar.success("API Key provided")
    else:
        st.sidebar.info("Enter API key to enable LLM features")
    
    st.sidebar.divider()
    
    # Chunking Strategy Selection
    st.sidebar.subheader("Chunking Strategy")
    chunking_strategy = st.sidebar.selectbox(
        "Select Chunking Method",
        options=list(CHUNKING_STRATEGIES.keys()),
        index=0
    )
    
    # Chunking Parameters
    chunk_size = st.sidebar.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=DEFAULT_CHUNK_SIZE,
        step=50
    )
    
    overlap = st.sidebar.slider(
        "Overlap (characters)",
        min_value=0,
        max_value=500,
        value=DEFAULT_OVERLAP,
        step=10
    )
    
    # Retriever Selection
    st.sidebar.subheader("Retriever Configuration")
    retriever_type = st.sidebar.selectbox(
        "Select Retriever",
        options=list(RETRIEVER_TYPES.keys()),
        index=0
    )
    
    top_k = st.sidebar.slider(
        "Top-K Results",
        min_value=1,
        max_value=20,
        value=DEFAULT_TOP_K,
        step=1
    )
    
    # Advanced Options
    with st.sidebar.expander("Advanced Chunking Options"):
        similarity_threshold = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05
        )
        
        sentences_per_chunk = st.slider(
            "Sentences per Chunk",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )
        
        num_topics = st.slider(
            "Number of Topics (Topic Chunking)",
            min_value=2,
            max_value=15,
            value=5,
            step=1
        )
    
    with st.sidebar.expander("Advanced Retrieval Options"):
        alpha = st.slider(
            "Hybrid Weight (BM25 vs Dense)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = Pure Dense, 1 = Pure BM25"
        )
        
        lambda_param = st.slider(
            "MMR Diversity Parameter",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0 = Max Diversity, 1 = Max Relevance"
        )
        
        initial_k = st.slider(
            "Initial Candidates (Reranking)",
            min_value=10,
            max_value=50,
            value=20,
            step=5
        )
        
        compare_strategies = st.checkbox("Compare All Strategies")
    
    return {
        "chunking_strategy": CHUNKING_STRATEGIES[chunking_strategy],
        "chunk_size": chunk_size,
        "overlap": overlap,
        "retriever_type": RETRIEVER_TYPES[retriever_type],
        "top_k": top_k,
        "similarity_threshold": similarity_threshold,
        "sentences_per_chunk": sentences_per_chunk,
        "num_topics": num_topics,
        "alpha": alpha,
        "lambda_param": lambda_param,
        "initial_k": initial_k,
        "compare_strategies": compare_strategies,
        "openai_api_key": openai_api_key
    }
