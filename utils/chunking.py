from typing import List, Dict, Any, Optional
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import re
import streamlit as st

class ChunkingStrategies:
    def __init__(self):
        self.model = None
    
    def get_embeddings_model(self):
        """Lazy load sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def fixed_length_chunking(self, text: str, chunk_size: int = 500) -> List[Dict[str, Any]]:
        """Split text into fixed-length chunks."""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=0,
            separator=" "
        )
        chunks = splitter.split_text(text)
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "fixed_length"}} 
                for i, chunk in enumerate(chunks)]
    
    def overlapping_chunking(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks."""
        splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separator=" "
        )
        chunks = splitter.split_text(text)
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "overlapping"}} 
                for i, chunk in enumerate(chunks)]
    
    def recursive_chunking(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """Split text using recursive character text splitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "recursive"}} 
                for i, chunk in enumerate(chunks)]
    
    def semantic_chunking(self, text: str, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Split text based on semantic similarity."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return [{"text": text, "metadata": {"chunk_id": 0, "strategy": "semantic"}}]
        
        model = self.get_embeddings_model()
        embeddings = model.encode(sentences)
        
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity >= similarity_threshold:
                current_chunk.append(sentences[i])
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i]]
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "semantic"}} 
                for i, chunk in enumerate(chunks)]
    
    def sentence_chunking(self, text: str, sentences_per_chunk: int = 3) -> List[Dict[str, Any]]:
        """Split text by sentences."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = " ".join(sentences[i:i + sentences_per_chunk])
            chunks.append(chunk)
        
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "sentence"}} 
                for i, chunk in enumerate(chunks)]
    
    def paragraph_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text by paragraphs."""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "paragraph"}} 
                for i, chunk in enumerate(paragraphs)]
    
    def heading_chunking(self, text: str) -> List[Dict[str, Any]]:
        """Split text by headings (Markdown/HTML style)."""
        # Look for markdown headings or numbered sections
        heading_patterns = [
            r'^#{1,6}\s+.+$',  # Markdown headings
            r'^\d+\.\s+.+$',   # Numbered sections
            r'^[A-Z][A-Z\s]+:',  # ALL CAPS headings
            r'^\*\*[^*]+\*\*',   # Bold headings
        ]
        
        chunks = []
        current_chunk = []
        current_heading = ""
        
        lines = text.split('\n')
        
        for line in lines:
            is_heading = any(re.match(pattern, line.strip(), re.MULTILINE) for pattern in heading_patterns)
            
            if is_heading and current_chunk:
                # Save previous chunk
                chunk_text = '\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "chunk_id": len(chunks),
                            "strategy": "heading",
                            "heading": current_heading
                        }
                    })
                current_chunk = [line]
                current_heading = line.strip()
            else:
                current_chunk.append(line)
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": len(chunks),
                        "strategy": "heading",
                        "heading": current_heading
                    }
                })
        
        return chunks if chunks else [{"text": text, "metadata": {"chunk_id": 0, "strategy": "heading"}}]
    
    def topic_chunking(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
        """Split text by topics using clustering."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) < num_topics:
            return [{"text": text, "metadata": {"chunk_id": 0, "strategy": "topic"}}]
        
        model = self.get_embeddings_model()
        embeddings = model.encode(sentences)
        
        # Use KMeans clustering to group sentences by topic
        kmeans = KMeans(n_clusters=min(num_topics, len(sentences)), random_state=42, n_init=10)
        clusters = kmeans.fit_predict(embeddings)
        
        # Group sentences by cluster
        topic_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in topic_groups:
                topic_groups[cluster_id] = []
            topic_groups[cluster_id].append(sentences[i])
        
        chunks = []
        for topic_id, topic_sentences in topic_groups.items():
            chunk_text = ". ".join(topic_sentences)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "chunk_id": len(chunks),
                    "strategy": "topic",
                    "topic_id": int(topic_id),
                    "sentence_count": len(topic_sentences)
                }
            })
        
        return chunks
    
    def dynamic_ai_chunking(self, text: str, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
        """Use AI to determine optimal chunk boundaries."""
        if not api_key:
            st.warning("⚠️ Dynamic AI chunking requires OpenAI API key")
            return self.recursive_chunking(text, 500, 50)  # Fallback
        
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # For very long texts, pre-chunk them
            if len(text) > 8000:
                pre_chunks = self.recursive_chunking(text, 2000, 200)
                final_chunks = []
                
                for pre_chunk in pre_chunks:
                    ai_chunks = self._ai_chunk_single(client, pre_chunk["text"])
                    for i, chunk in enumerate(ai_chunks):
                        final_chunks.append({
                            "text": chunk,
                            "metadata": {
                                "chunk_id": len(final_chunks),
                                "strategy": "dynamic_ai",
                                "parent_chunk": pre_chunk["metadata"]["chunk_id"]
                            }
                        })
                return final_chunks
            else:
                ai_chunks = self._ai_chunk_single(client, text)
                return [{"text": chunk, "metadata": {"chunk_id": i, "strategy": "dynamic_ai"}} 
                       for i, chunk in enumerate(ai_chunks)]
                
        except Exception as e:
            st.error(f"AI chunking failed: {str(e)}")
            return self.recursive_chunking(text, 500, 50)  # Fallback
    
    def _ai_chunk_single(self, client, text: str) -> List[str]:
        """Helper method to chunk a single text using AI."""
        prompt = f"""Analyze the following text and split it into logical, coherent chunks. Each chunk should:
1. Be semantically complete (contain complete thoughts/concepts)
2. Be 200-800 words long
3. Have natural boundaries (don't cut sentences)
4. Maintain context within each chunk

Text to chunk:
{text}

Return only the chunks separated by "---CHUNK---" markers. Do not include explanations."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at text segmentation and chunking for information retrieval."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        chunks = [chunk.strip() for chunk in result.split("---CHUNK---") if chunk.strip()]
        
        return chunks if chunks else [text]
    
    def chunk_text(self, text: str, strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """Main method to chunk text based on strategy."""
        if strategy == "fixed_length":
            return self.fixed_length_chunking(text, kwargs.get('chunk_size', 500))
        elif strategy == "overlapping":
            return self.overlapping_chunking(
                text, 
                kwargs.get('chunk_size', 500), 
                kwargs.get('overlap', 50)
            )
        elif strategy == "recursive":
            return self.recursive_chunking(
                text, 
                kwargs.get('chunk_size', 500), 
                kwargs.get('overlap', 50)
            )
        elif strategy == "semantic":
            return self.semantic_chunking(text, kwargs.get('similarity_threshold', 0.7))
        elif strategy == "sentence":
            return self.sentence_chunking(text, kwargs.get('sentences_per_chunk', 3))
        elif strategy == "paragraph":
            return self.paragraph_chunking(text)
        elif strategy == "heading":
            return self.heading_chunking(text)
        elif strategy == "topic":
            return self.topic_chunking(text, kwargs.get('num_topics', 5))
        elif strategy == "dynamic_ai":
            return self.dynamic_ai_chunking(text, kwargs.get('api_key'))
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
