#!/usr/bin/env python3
"""
ColBERT Retrieval Implementation
MaxSim-based ranking with FAISS nearest neighbor search.
"""

import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging

try:
    from .colbert_model import ColBERT
    from .colbert_indexing import ColBERTIndexer
except ImportError:
    from colbert_model import ColBERT
    from colbert_indexing import ColBERTIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColBERTRetriever:
    """
    Retrieves documents using ColBERT MaxSim scoring.
    """
    
    def __init__(
        self,
        model: ColBERT,
        indexer: ColBERTIndexer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize retriever.
        
        Args:
            model: Trained ColBERT model
            indexer: ColBERTIndexer with built FAISS index
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.indexer = indexer
        self.device = device
        
        # Pre-compute embedding-to-document lookup for fast retrieval
        logger.info("Building embedding-to-document lookup table...")
        self._build_embedding_lookup()
        
        logger.info(f"Retriever initialized")
        logger.info(f"  Documents: {len(indexer.doc_ids):,}")
        logger.info(f"  Embeddings: {sum(indexer.doc_lengths):,}")
    
    def _build_embedding_lookup(self):
        """
        Build a lookup table: embedding_idx -> doc_idx
        This avoids O(n) linear search for each embedding lookup.
        """
        total_embeddings = sum(self.indexer.doc_lengths)
        self.emb_to_doc = np.empty(total_embeddings, dtype=np.int32)
        
        emb_idx = 0
        for doc_idx, doc_len in enumerate(self.indexer.doc_lengths):
            self.emb_to_doc[emb_idx:emb_idx + doc_len] = doc_idx
            emb_idx += doc_len
        
        logger.info(f"  Built lookup table for {total_embeddings:,} embeddings")
    
    def retrieve(
        self,
        queries: List[str],
        k: int = 100,
        k_nn: int = 10,  # Top-k nearest neighbors per query token
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Retrieve top-k documents for each query using MaxSim.
        
        Args:
            queries: List of query strings
            k: Number of documents to retrieve
            k_nn: Number of nearest neighbors to consider per query token
            batch_size: Batch size for query encoding
            show_progress: Show progress bar
        
        Returns:
            results: {query_text: [(doc_id, score), ...]}
        """
        logger.info(f"Retrieving for {len(queries)} queries...")
        logger.info(f"  Top-k documents: {k}")
        logger.info(f"  Top-k_nn per token: {k_nn}")
        
        results = {}
        
        # Process queries in batches
        num_batches = (len(queries) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for i in tqdm(
                range(num_batches),
                desc="Retrieving",
                disable=not show_progress
            ):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(queries))
                
                batch_queries = queries[start_idx:end_idx]
                
                # Encode queries: [batch_size, Nq, dim]
                query_embeddings = self.model.query(batch_queries)
                
                # Retrieve for each query in batch
                for j, query_text in enumerate(batch_queries):
                    query_emb = query_embeddings[j]  # [Nq, dim]
                    
                    # Retrieve top-k documents
                    doc_scores = self._maxsim_retrieve(query_emb, k, k_nn)
                    
                    results[query_text] = doc_scores
        
        return results
    
    def _maxsim_retrieve(
        self,
        query_emb: torch.Tensor,
        k: int,
        k_nn: int
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for a single query using MaxSim.
        
        MaxSim strategy:
        1. For each query token, find top-k_nn nearest document tokens using FAISS
        2. Aggregate scores at document level (sum of max similarities)
        3. Return top-k documents by score
        
        Args:
            query_emb: [Nq, dim] query embeddings
            k: Number of documents to retrieve
            k_nn: Number of nearest neighbors per query token
        
        Returns:
            List of (doc_id, score) tuples
        """
        # Move query to CPU and normalize
        query_emb_np = query_emb.cpu().numpy()
        faiss.normalize_L2(query_emb_np)
        
        Nq = query_emb_np.shape[0]
        
        # Search FAISS for each query token: [Nq, k_nn]
        similarities, indices = self.indexer.faiss_index.search(query_emb_np, k_nn)
        
        # Map embedding indices to document indices and aggregate scores
        doc_scores = {}  # doc_id -> total score (sum of max similarities)
        
        for q_idx in range(Nq):
            # Get top-k_nn embeddings for this query token
            token_scores = similarities[q_idx]  # [k_nn]
            token_indices = indices[q_idx]      # [k_nn]
            
            # For this query token, find the MAXIMUM similarity to each document
            query_token_doc_max = {}  # doc_id -> max similarity for THIS query token
            
            for emb_idx, sim_score in zip(token_indices, token_scores):
                if emb_idx < 0 or emb_idx >= len(self.emb_to_doc):  # FAISS returns -1 for invalid indices
                    continue
                
                # Fast O(1) lookup: embedding -> document
                doc_idx = self.emb_to_doc[emb_idx]
                doc_id = self.indexer.doc_ids[doc_idx]
                
                # MaxSim: For this query token, keep the MAXIMUM similarity to this document
                if doc_id not in query_token_doc_max:
                    query_token_doc_max[doc_id] = sim_score
                else:
                    query_token_doc_max[doc_id] = max(query_token_doc_max[doc_id], sim_score)
            
            # Now add the max similarity for this query token to each document's total score
            for doc_id, max_sim in query_token_doc_max.items():
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0.0
                doc_scores[doc_id] += max_sim  # Sum of maxes (correct MaxSim!)
        
        # Sort by score and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:k]
    
    def rerank(
        self,
        queries: List[str],
        candidates: Dict[str, List[str]],  # {query: [doc_ids]}
        batch_size: int = 32
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Re-rank candidate documents using full MaxSim scoring.
        
        Args:
            queries: List of query strings
            candidates: {query: [doc_ids]} candidate documents per query
            batch_size: Batch size for encoding
        
        Returns:
            results: {query: [(doc_id, score), ...]}
        """
        logger.info(f"Re-ranking for {len(queries)} queries...")
        
        results = {}
        
        with torch.no_grad():
            for query in tqdm(queries, desc="Re-ranking"):
                if query not in candidates:
                    results[query] = []
                    continue
                
                # Encode query
                query_emb = self.model.query([query])[0]  # [Nq, dim]
                
                # Get candidate documents
                cand_ids = candidates[query]
                
                # Find document indices
                doc_indices = [
                    self.indexer.doc_ids.index(doc_id)
                    for doc_id in cand_ids
                    if doc_id in self.indexer.doc_ids
                ]
                
                # Compute scores
                scores = []
                for doc_idx in doc_indices:
                    # Get document embeddings
                    doc_emb = self.indexer.get_document_embeddings(doc_idx)
                    doc_emb_tensor = torch.from_numpy(doc_emb).to(self.device).unsqueeze(0)
                    query_emb_batch = query_emb.unsqueeze(0)
                    
                    # Compute MaxSim score
                    score = self.model.score(query_emb_batch, doc_emb_tensor).item()
                    scores.append((self.indexer.doc_ids[doc_idx], score))
                
                # Sort by score
                scores.sort(key=lambda x: x[1], reverse=True)
                results[query] = scores
        
        return results


if __name__ == "__main__":
    """Test retrieval with dummy data."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    print("Testing ColBERT retrieval...")
    
    # Create model
    model = ColBERT()
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create and load index
    indexer = ColBERTIndexer(
        model=model,
        index_path="indices/test_index",
        batch_size=10
    )
    
    if not (Path("indices/test_index") / "config.json").exists():
        print("Creating test index...")
        # Create dummy documents
        documents = [
            (f"doc_{i}", f"Document {i}: " + " ".join([f"word{j}" for j in range(10)]))
            for i in range(100)
        ]
        indexer.index_documents(documents)
        indexer.build_faiss_index(use_gpu=torch.cuda.is_available())
        indexer.save()
    else:
        print("Loading existing test index...")
        indexer.load(use_gpu=torch.cuda.is_available())
    
    # Create retriever
    retriever = ColBERTRetriever(
        model=model,
        indexer=indexer
    )
    
    # Test retrieval
    print("\nTesting retrieval...")
    queries = ["sample query about documents", "another test query"]
    results = retriever.retrieve(queries, k=10, k_nn=5)
    
    for query, docs in results.items():
        print(f"\nQuery: {query}")
        print(f"Top-3 results:")
        for doc_id, score in docs[:3]:
            print(f"  {doc_id}: {score:.4f}")
    
    print("\n✅ Retrieval test completed!")
