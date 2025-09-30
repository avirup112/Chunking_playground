import streamlit as st
from typing import List, Dict, Any, Optional
from groq import Groq 

class LLMManager:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.client = None
        if api_key:
            self.client = Groq(api_key=api_key)
    
    def is_available(self) -> bool:
        """Check if LLM is available (API key provided)."""
        return self.client is not None
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]], model: str = "llama-3.1-8b-instant") -> str:
        """Generate answer using LLM with retrieved chunks as context."""
        if not self.is_available():
            return "API key not provided. Please enter your API key in the sidebar."
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"Chunk {i+1}: {chunk['text']}" 
            for i, chunk in enumerate(retrieved_chunks[:5])  # Use top 5 chunks
        ])
        
        # Create prompt
        prompt = f"""Based on the following context from a document, please answer the user's question.

Context:
{context}

Question: {query}

Please provide a clear and concise answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def compare_answers(self, query: str, strategies_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """Generate answers using different retrieval strategies for comparison."""
        if not self.is_available():
            return {"error": "OpenAI API key not provided"}
        
        answers = {}
        for strategy, chunks in strategies_results.items():
            if chunks:
                answer = self.generate_answer(query, chunks)
                answers[strategy] = answer
        
        return answers

def display_llm_answer(answer: str, retrieved_chunks: List[Dict[str, Any]]):
    """Display LLM answer with source chunks."""
    st.subheader("🤖 AI-Generated Answer")
    
    # Display the answer
    st.markdown("### Answer:")
    st.write(answer)
    
    # Display source chunks
    if retrieved_chunks:
        st.markdown("### Source Chunks:")
        for i, chunk in enumerate(retrieved_chunks[:3]):  # Show top 3 source chunks
            with st.expander(f"Source {i+1} (Score: {chunk.get('score', 0):.3f})"):
                st.text_area(
                    f"Chunk {i+1}",
                    chunk["text"],
                    height=100,
                    key=f"llm_source_{i}",
                    disabled=True
                )

def display_answer_comparison(answers: Dict[str, str], query: str):
    """Display comparison of answers from different strategies."""
    if not answers or "error" in answers:
        st.error(answers.get("error", "No answers to compare"))
        return
    
    st.subheader("🔄 Answer Comparison Across Strategies")
    st.markdown(f"**Question:** {query}")
    
    for strategy, answer in answers.items():
        with st.expander(f"Answer using {strategy.upper()} retrieval"):
            st.write(answer)