from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import streamlit as st
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class RetrieverManager:
    def __init__(self, vectorstore_dir: Path):
        self.vectorstore_dir = vectorstore_dir
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker_model = None  # Lazy load
        self.bm25 = None
        self.faiss_index = None
        self.chroma_client = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunks = []
        self.chunk_embeddings = None
        
    def setup_bm25(self, chunks: List[Dict[str, Any]]):
        """Setup BM25 retriever."""
        corpus = [chunk["text"].split() for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunks = chunks
    
    def setup_faiss(self, chunks: List[Dict[str, Any]]):
        """Setup FAISS retriever."""
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.faiss_index.add(embeddings.astype('float32'))
        self.chunks = chunks
    
    def setup_chroma(self, chunks: List[Dict[str, Any]], collection_name: str = "documents"):
        """Setup Chroma retriever."""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vectorstore_dir / "chroma")
            )
            
            # Delete existing collection if it exists
            try:
                self.chroma_client.delete_collection(collection_name)
            except:
                pass
            
            collection = self.chroma_client.create_collection(collection_name)
            
            texts = [chunk["text"] for chunk in chunks]
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            collection.add(
                documents=texts,
                ids=ids,
                metadatas=metadatas
            )
            
            self.chunks = chunks
            return True
        except Exception as e:
            st.error(f"Error setting up Chroma: {str(e)}")
            return False
    
    def retrieve_bm25(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using BM25."""
        if self.bm25 is None:
            return []
        
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["score"] = float(scores[idx])
                results.append(result)
        
        return results
    
    def retrieve_faiss(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve using FAISS."""
        if self.faiss_index is None:
            return []
        
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def retrieve_chroma(self, query: str, top_k: int = 5, collection_name: str = "documents") -> List[Dict[str, Any]]:
        """Retrieve using Chroma."""
        if self.chroma_client is None:
            return []
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            retrieved_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                result = {
                    "text": doc,
                    "metadata": metadata,
                    "score": 1 - distance  # Convert distance to similarity
                }
                retrieved_results.append(result)
            
            return retrieved_results
        except Exception as e:
            st.error(f"Error retrieving from Chroma: {str(e)}")
            return []
    
    def setup_hybrid(self, chunks: List[Dict[str, Any]]):
        """Setup hybrid retriever (BM25 + Dense)."""
        # Setup both BM25 and FAISS
        self.setup_bm25(chunks)
        self.setup_faiss(chunks)
        
        # Setup TF-IDF for additional keyword matching
        texts = [chunk["text"] for chunk in chunks]
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
    
    def retrieve_hybrid(self, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining BM25 and dense retrieval."""
        if self.bm25 is None or self.faiss_index is None:
            return []
        
        # Get results from both retrievers
        bm25_results = self.retrieve_bm25(query, top_k * 2)
        faiss_results = self.retrieve_faiss(query, top_k * 2)
        
        # Normalize scores
        if bm25_results:
            max_bm25 = max(r["score"] for r in bm25_results)
            for r in bm25_results:
                r["score"] = r["score"] / max_bm25 if max_bm25 > 0 else 0
        
        if faiss_results:
            max_faiss = max(r["score"] for r in faiss_results)
            for r in faiss_results:
                r["score"] = r["score"] / max_faiss if max_faiss > 0 else 0
        
        # Combine scores
        combined_scores = {}
        for result in bm25_results:
            chunk_id = result["metadata"]["chunk_id"]
            combined_scores[chunk_id] = alpha * result["score"]
        
        for result in faiss_results:
            chunk_id = result["metadata"]["chunk_id"]
            if chunk_id in combined_scores:
                combined_scores[chunk_id] += (1 - alpha) * result["score"]
            else:
                combined_scores[chunk_id] = (1 - alpha) * result["score"]
        
        # Sort and return top-k
        sorted_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for chunk_id, score in sorted_chunks:
            if chunk_id < len(self.chunks):
                result = self.chunks[chunk_id].copy()
                result["score"] = float(score)
                results.append(result)
        
        return results
    
    def retrieve_mmr(self, query: str, top_k: int = 5, lambda_param: float = 0.5) -> List[Dict[str, Any]]:
        """Maximal Marginal Relevance retrieval for diversity."""
        if self.faiss_index is None:
            return []
        
        # Get initial candidates (more than needed)
        initial_k = min(top_k * 3, len(self.chunks))
        candidates = self.retrieve_faiss(query, initial_k)
        
        if not candidates:
            return []
        
        # MMR algorithm
        selected = []
        remaining = candidates.copy()
        
        # Select first document (highest relevance)
        selected.append(remaining.pop(0))
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score (already computed)
                relevance = candidate["score"]
                
                # Compute max similarity to already selected documents
                max_sim = 0
                candidate_embedding = self.model.encode([candidate["text"]])
                
                for selected_doc in selected:
                    selected_embedding = self.model.encode([selected_doc["text"]])
                    similarity = cosine_similarity(candidate_embedding, selected_embedding)[0][0]
                    max_sim = max(max_sim, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((candidate, mmr_score))
            
            # Select document with highest MMR score
            best_candidate, _ = max(mmr_scores, key=lambda x: x[1])
            selected.append(best_candidate)
            remaining.remove(best_candidate)
        
        return selected
    
    def retrieve_reranking(self, query: str, top_k: int = 5, initial_k: int = 20) -> List[Dict[str, Any]]:
        """Two-stage retrieval with reranking."""
        # First stage: get initial candidates
        initial_results = self.retrieve_faiss(query, initial_k)
        
        if not initial_results:
            return []
        
        # Second stage: rerank using cross-encoder (simulated with sentence similarity)
        reranked_results = []
        query_embedding = self.model.encode([query])
        
        for result in initial_results:
            chunk_embedding = self.model.encode([result["text"]])
            # Use more sophisticated similarity for reranking
            similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
            
            # Combine original score with reranking score
            combined_score = 0.3 * result["score"] + 0.7 * similarity
            result["score"] = float(combined_score)
            reranked_results.append(result)
        
        # Sort by new scores and return top-k
        reranked_results.sort(key=lambda x: x["score"], reverse=True)
        return reranked_results[:top_k]
    
    def retrieve_multi_vector(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Multi-vector retrieval using different representations."""
        if not self.chunks:
            return []
        
        # Create multiple representations for each chunk
        chunk_representations = []
        for chunk in self.chunks:
            text = chunk["text"]
            
            # Original text
            original_emb = self.model.encode([text])[0]
            
            # Summary (first sentence)
            sentences = text.split('.')
            summary = sentences[0] if sentences else text[:100]
            summary_emb = self.model.encode([summary])[0]
            
            # Keywords (simple extraction)
            words = text.lower().split()
            keywords = [w for w in words if len(w) > 5][:10]  # Simple keyword extraction
            keywords_text = " ".join(keywords)
            keywords_emb = self.model.encode([keywords_text])[0] if keywords else original_emb
            
            chunk_representations.append({
                "chunk": chunk,
                "original": original_emb,
                "summary": summary_emb,
                "keywords": keywords_emb
            })
        
        # Query against all representations
        query_emb = self.model.encode([query])[0]
        
        scores = []
        for i, repr_dict in enumerate(chunk_representations):
            # Compute similarities for each representation
            orig_sim = cosine_similarity([query_emb], [repr_dict["original"]])[0][0]
            summ_sim = cosine_similarity([query_emb], [repr_dict["summary"]])[0][0]
            key_sim = cosine_similarity([query_emb], [repr_dict["keywords"]])[0][0]
            
            # Weighted combination
            combined_score = 0.5 * orig_sim + 0.3 * summ_sim + 0.2 * key_sim
            scores.append((i, combined_score))
        
        # Sort and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in scores[:top_k]:
            result = chunk_representations[i]["chunk"].copy()
            result["score"] = float(score)
            results.append(result)
        
        return results
    
    def retrieve(self, query: str, retriever_type: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Main retrieval method."""
        if retriever_type == "bm25":
            return self.retrieve_bm25(query, top_k)
        elif retriever_type == "faiss":
            return self.retrieve_faiss(query, top_k)
        elif retriever_type == "chroma":
            return self.retrieve_chroma(query, top_k)
        elif retriever_type == "hybrid":
            return self.retrieve_hybrid(query, top_k, kwargs.get('alpha', 0.5))
        elif retriever_type == "mmr":
            return self.retrieve_mmr(query, top_k, kwargs.get('lambda_param', 0.5))
        elif retriever_type == "reranking":
            return self.retrieve_reranking(query, top_k, kwargs.get('initial_k', 20))
        elif retriever_type == "multi_vector":
            return self.retrieve_multi_vector(query, top_k)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")
