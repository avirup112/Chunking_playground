import streamlit as st
import os
from pathlib import Path
# Import custom modules
from config import UPLOADS_DIR, VECTORSTORE_DIR, SUPPORTED_EXTENSIONS
from utils.file_loader import extract_text_from_file, save_uploaded_file
from utils.chunking import ChunkingStrategies
from utils.retrievers import RetrieverManager
from utils.llm_integration import LLMManager, display_llm_answer, display_answer_comparison
from components.sidebar import render_sidebar
from components.display import (
    display_document_info,
    display_chunking_results,
    display_query_interface,
    display_retrieval_interface,
    display_comparison_interface,
    display_educational_content
)

# Page configuration
st.set_page_config(
    page_title="Chunking Playground",
    page_icon="🔪",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    st.title("🔪 Chunking Playground")
    st.markdown("**Experiment with different chunking strategies and retrieval methods for RAG pipelines**")
    
    # Initialize session state
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "filename" not in st.session_state:
        st.session_state.filename = ""
    
    # Render sidebar
    config = render_sidebar()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📁 Document Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload a PDF, TXT, or DOCX file to analyze"
        )
        
        if uploaded_file is not None:
            # Save and process file
            file_path = save_uploaded_file(uploaded_file, UPLOADS_DIR)
            
            with st.spinner("Extracting text..."):
                text = extract_text_from_file(file_path)
            
            if text:
                st.session_state.document_text = text
                st.session_state.filename = uploaded_file.name
                st.success("✅ File processed successfully!")
            else:
                st.error("❌ Failed to extract text from file")
    
    with col2:
        if st.session_state.document_text:
            display_document_info(st.session_state.document_text, st.session_state.filename)
    
    # Process document if available
    if st.session_state.document_text:
        st.divider()
        
        # Initialize chunking and retrieval
        chunker = ChunkingStrategies()
        retriever = RetrieverManager(VECTORSTORE_DIR)
        
        # Generate chunks
        with st.spinner("Generating chunks..."):
            chunks = chunker.chunk_text(
                st.session_state.document_text,
                config["chunking_strategy"],
                chunk_size=config["chunk_size"],
                overlap=config["overlap"],
                similarity_threshold=config["similarity_threshold"],
                sentences_per_chunk=config["sentences_per_chunk"],
                num_topics=config["num_topics"],
                api_key=config.get("openai_api_key")
            )
            st.session_state.chunks = chunks
        
        # Display chunking results
        display_chunking_results(chunks, config["chunking_strategy"])
        
        st.divider()
        
        # Setup retriever
        with st.spinner(f"Setting up {config['retriever_type'].upper()} retriever..."):
            if config["retriever_type"] == "bm25":
                retriever.setup_bm25(chunks)
            elif config["retriever_type"] == "faiss":
                retriever.setup_faiss(chunks)
            elif config["retriever_type"] == "chroma":
                retriever.setup_chroma(chunks)
            elif config["retriever_type"] == "hybrid":
                retriever.setup_hybrid(chunks)
            elif config["retriever_type"] in ["mmr", "reranking", "multi_vector"]:
                retriever.setup_faiss(chunks)  # These use FAISS as base
        
        # Initialize LLM manager
        llm_manager = LLMManager(config.get("openai_api_key"))
        
        # Query interface
        query, search_button, llm_button = display_query_interface()
        
        if (search_button or llm_button) and query:
            # Perform retrieval
            with st.spinner("Retrieving relevant chunks..."):
                results = retriever.retrieve(
                    query,
                    config["retriever_type"],
                    config["top_k"],
                    alpha=config["alpha"],
                    lambda_param=config["lambda_param"],
                    initial_k=config["initial_k"]
                )
            
            # Display results
            display_retrieval_interface(results, query, config["retriever_type"])
            
            # LLM Answer Generation
            if llm_button and llm_manager.is_available():
                st.divider()
                with st.spinner("Generating AI answer..."):
                    answer = llm_manager.generate_answer(query, results)
                display_llm_answer(answer, results)
            elif llm_button and not llm_manager.is_available():
                st.warning("⚠️ Please enter your OpenAI API key in the sidebar to use LLM features.")
            
            # Comparison mode
            if config["compare_strategies"]:
                st.divider()
                
                with st.spinner("Comparing all retrieval strategies..."):
                    comparison_results = {}
                    
                    # Test all retrievers
                    retriever_tests = [
                        ("BM25", "bm25"), 
                        ("FAISS", "faiss"), 
                        ("Chroma", "chroma"),
                        ("Hybrid", "hybrid"),
                        ("MMR", "mmr"),
                        ("Reranking", "reranking"),
                        ("Multi-Vector", "multi_vector")
                    ]
                    
                    for ret_name, ret_type in retriever_tests:
                        try:
                            if ret_type == "bm25":
                                retriever.setup_bm25(chunks)
                            elif ret_type == "faiss":
                                retriever.setup_faiss(chunks)
                            elif ret_type == "chroma":
                                retriever.setup_chroma(chunks)
                            elif ret_type == "hybrid":
                                retriever.setup_hybrid(chunks)
                            elif ret_type in ["mmr", "reranking", "multi_vector"]:
                                retriever.setup_faiss(chunks)  # These use FAISS as base
                            
                            comp_results = retriever.retrieve(
                                query, 
                                ret_type, 
                                config["top_k"],
                                alpha=config["alpha"],
                                lambda_param=config["lambda_param"],
                                initial_k=config["initial_k"]
                            )
                            comparison_results[ret_name] = comp_results
                        except Exception as e:
                            st.warning(f"Could not test {ret_name}: {str(e)}")
                
                display_comparison_interface(comparison_results, query)
                
                # LLM comparison if API key is available
                if llm_button and llm_manager.is_available():
                    st.divider()
                    with st.spinner("Generating answers for all strategies..."):
                        comparison_answers = llm_manager.compare_answers(query, comparison_results)
                    display_answer_comparison(comparison_answers, query)
            
            # Educational content
            st.divider()
            display_educational_content(
                chunks, 
                results if 'results' in locals() else [], 
                config["chunking_strategy"], 
                config["retriever_type"]
            )
    
    else:
        st.info("👆 Upload a document to get started!")
        
        # Show example/demo content
        with st.expander("ℹ️ How to use this playground"):
            st.markdown("""
            ### Getting Started
            1. **Upload a document** (PDF, TXT, or DOCX) using the file uploader
            2. **Choose a chunking strategy** from the sidebar
            3. **Adjust parameters** like chunk size and overlap
            4. **Select a retriever** (BM25, FAISS, Chroma, Hybrid, MMR, Reranking, Multi-Vector)
            5. **Enter a query** to test retrieval performance
            6. **Compare strategies** by enabling comparison mode
            
            ### Chunking Strategies
            - **Fixed Length**: Splits text into equal-sized chunks
            - **Overlapping (Sliding Window)**: Adds overlap between chunks to preserve context
            - **Recursive**: Respects natural text boundaries (paragraphs, sentences)
            - **Semantic**: Groups semantically similar content using embeddings
            - **Sentence**: Splits by complete sentences
            - **Paragraph**: Splits by paragraphs
            - **Heading Based**: Splits by document headings and sections
            - **Topic Based**: Uses clustering to group similar content
            - **Dynamic AI**: Uses AI to determine optimal chunk boundaries
            
            ### Retrievers
            - **BM25 (Keyword)**: Classical keyword-based retrieval using TF-IDF
            - **FAISS (Dense)**: Fast vector similarity search using embeddings
            - **Chroma (Dense)**: Persistent vector database with metadata support
            - **Hybrid**: Combines BM25 keyword + dense semantic search
            - **MMR (Diverse)**: Maximal Marginal Relevance for diverse results
            - **Reranking**: Two-stage retrieval with sophisticated reranking
            - **Multi-Vector**: Uses multiple representations per chunk
            """)

if __name__ == "__main__":
    main()
