import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import List, Dict, Any
import numpy as np

def display_chunk_statistics(chunks: List[Dict[str, Any]]):
    """Display statistics about chunks."""
    if not chunks:
        return
    
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", len(chunks))
    
    with col2:
        st.metric("Avg Chunk Size", f"{np.mean(chunk_lengths):.0f} chars")
    
    with col3:
        st.metric("Min Chunk Size", f"{min(chunk_lengths)} chars")
    
    with col4:
        st.metric("Max Chunk Size", f"{max(chunk_lengths)} chars")
    
    # Chunk size distribution
    fig = px.histogram(
        x=chunk_lengths,
        nbins=20,
        title="Chunk Size Distribution",
        labels={"x": "Chunk Size (characters)", "y": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_chunks_with_highlighting(chunks: List[Dict[str, Any]], max_display: int = 10, key_prefix: str = "main"):
    """Display chunks with syntax highlighting."""
    st.subheader(f"Generated Chunks (showing first {min(len(chunks), max_display)})")
    
    for i, chunk in enumerate(chunks[:max_display]):
        with st.expander(f"Chunk {i+1} ({len(chunk['text'])} chars)"):
            st.text_area(
                f"Chunk {i+1}",
                chunk["text"],
                height=150,
                key=f"{key_prefix}_chunk_{i}",
                disabled=True
            )
            
            # Show metadata
            if "metadata" in chunk:
                st.json(chunk["metadata"])

def display_retrieval_results(results: List[Dict[str, Any]], query: str, key_prefix: str = "main"):
    """Display retrieval results with scores."""
    if not results:
        st.warning("No results found for the query.")
        return
    
    st.subheader(f"Retrieved Results for: '{query}'")
    
    for i, result in enumerate(results):
        score = result.get("score", 0)
        
        with st.expander(f"Result {i+1} (Score: {score:.3f})"):
            st.text_area(
                f"Retrieved Chunk {i+1}",
                result["text"],
                height=100,
                key=f"{key_prefix}_result_{i}",
                disabled=True
            )
            
            # Show metadata and score
            metadata = result.get("metadata", {})
            metadata["retrieval_score"] = score
            st.json(metadata)

def compare_retrieval_strategies(results_dict: Dict[str, List[Dict[str, Any]]]):
    """Compare different retrieval strategies."""
    if not results_dict:
        return
    
    st.subheader("Retrieval Strategy Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for strategy, results in results_dict.items():
        if results:
            avg_score = np.mean([r.get("score", 0) for r in results])
            num_results = len(results)
            comparison_data.append({
                "Strategy": strategy,
                "Avg Score": avg_score,
                "Results Count": num_results
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Bar chart comparing average scores
        fig = px.bar(
            df,
            x="Strategy",
            y="Avg Score",
            title="Average Retrieval Scores by Strategy",
            color="Avg Score",
            color_continuous_scale="viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table
        st.dataframe(df, use_container_width=True)

def display_chunking_strategy_comparison(strategies_results: Dict[str, List[Dict[str, Any]]]):
    """Compare different chunking strategies."""
    if not strategies_results:
        return
    
    st.subheader("Chunking Strategy Comparison")
    
    comparison_data = []
    for strategy, chunks in strategies_results.items():
        if chunks:
            chunk_lengths = [len(chunk["text"]) for chunk in chunks]
            comparison_data.append({
                "Strategy": strategy,
                "Total Chunks": len(chunks),
                "Avg Chunk Size": np.mean(chunk_lengths),
                "Min Size": min(chunk_lengths),
                "Max Size": max(chunk_lengths),
                "Std Dev": np.std(chunk_lengths)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        
        for _, row in df.iterrows():
            fig.add_trace(go.Bar(
                name=row["Strategy"],
                x=["Total Chunks", "Avg Size", "Min Size", "Max Size"],
                y=[row["Total Chunks"], row["Avg Chunk Size"], row["Min Size"], row["Max Size"]]
            ))
        
        fig.update_layout(
            title="Chunking Strategy Metrics Comparison",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
