"""
Efficient FAISS-based index for ConstBERT with exact MaxSim computation.
This is an engineering optimization - the MaxSim computation remains identical to the paper.
"""
import numpy as np
import faiss
from typing import List, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm
import time


class ConstBERTFAISSIndex:
    """
    FAISS-based index for efficient ConstBERT retrieval.
    Uses FAISS for fast nearest neighbor search, then computes exact MaxSim.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize FAISS index.
        
        Args:
            use_gpu: Whether to use GPU for FAISS (if available)
        """
        self.doc_ids = []
        self.embeddings = None  # (num_docs, C, dim)
        self.faiss_index = None
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray):
        """
        Add documents to the index.
        
        Args:
            doc_ids: List of document IDs
            embeddings: Document embeddings (num_docs, C, dim)
        """
        assert len(doc_ids) == len(embeddings)
        
        self.doc_ids.extend(doc_ids)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)
        
        # Build FAISS index on flattened vectors for fast candidate retrieval
        num_docs, C, dim = self.embeddings.shape
        flat_embeddings = self.embeddings.reshape(num_docs * C, dim).astype('float32')
        
        print(f"Building FAISS index for {num_docs} documents...")
        
        # Use IVF (Inverted File) index for large-scale retrieval (much faster)
        if num_docs > 100000:
            # IVF with nlist clusters (fewer clusters = faster training)
            nlist = 4096  # Fixed reasonable value for 8M+ docs
            quantizer = faiss.IndexFlatIP(dim)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train on sample (not all vectors) - MUCH faster, negligible accuracy loss
            train_size = min(len(flat_embeddings), 10_000_000)  # 10M vectors max
            print(f"Training IVF index with {nlist} clusters on {train_size:,} sample vectors...")
            train_sample = flat_embeddings[np.random.choice(len(flat_embeddings), train_size, replace=False)]
            self.faiss_index.train(train_sample)
            
            self.faiss_index.nprobe = 128  # Search 128 clusters (balance speed/accuracy)
            print(f"Index trained, nprobe={self.faiss_index.nprobe}")
        else:
            # Use flat index for small collections
            self.faiss_index = faiss.IndexFlatIP(dim)
        
        if self.use_gpu and faiss.get_num_gpus() > 0:
            print("Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        print(f"Adding {len(flat_embeddings):,} vectors to index...")
        self.faiss_index.add(flat_embeddings)
        print(f"FAISS index built: {self.faiss_index.ntotal:,} vectors")
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 1000,
        candidate_mult: int = 10,
        show_progress: bool = False
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for top-k documents using FAISS + exact MaxSim.
        
        Strategy:
        1. Use FAISS to find top candidate_mult*k candidate DOCUMENTS (fast)
        2. Compute exact MaxSim on candidates only (accurate)
        3. Return top-k by exact MaxSim score
        
        Args:
            query_embeddings: Query embeddings (batch_size, num_tokens, dim) or (num_tokens, dim)
            k: Number of results to return per query
            candidate_mult: Retrieve this many candidates per doc for reranking
            show_progress: Show progress bar for candidate retrieval
        
        Returns:
            Tuple of (doc_ids, scores) where each is a list of lists
        """
        # Handle single query case
        if query_embeddings.ndim == 2:
            query_embeddings = query_embeddings[np.newaxis, :]
        
        batch_size = query_embeddings.shape[0]
        num_docs, C, dim = self.embeddings.shape
        
        # Determine number of candidates
        num_candidates = min(k * candidate_mult, num_docs)
        
        all_doc_ids = []
        all_scores = []
        
        iterator = range(batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="FAISS search + MaxSim")
        
        for i in iterator:
            query_emb = query_embeddings[i].astype('float32')  # (num_tokens, dim)
            num_tokens = query_emb.shape[0]
            
            # Step 1: FAISS search to get candidate document IDs (very fast)
            # Search for top vectors per query token, then deduplicate to get unique doc IDs
            # Use fewer vectors per token to control total search size
            vectors_per_token = max(100, num_candidates // max(1, num_tokens // 2))
            _, vector_indices = self.faiss_index.search(query_emb, vectors_per_token)  # (num_tokens, vectors_per_token)
            
            # Map vector indices to document indices and deduplicate
            candidate_doc_indices = np.unique(vector_indices.flatten() // C)
            candidate_doc_indices = candidate_doc_indices[candidate_doc_indices < num_docs]
            
            # Limit to num_candidates documents
            if len(candidate_doc_indices) > num_candidates:
                candidate_doc_indices = candidate_doc_indices[:num_candidates]
            
            # Step 2: Compute exact MaxSim on candidates (batched for efficiency)
            batch_size = 500  # Process candidates in batches to balance speed and memory
            scores = np.zeros(len(candidate_doc_indices))
            
            for start_idx in range(0, len(candidate_doc_indices), batch_size):
                end_idx = min(start_idx + batch_size, len(candidate_doc_indices))
                batch_indices = candidate_doc_indices[start_idx:end_idx]
                
                # Vectorized MaxSim for this batch
                batch_embs = self.embeddings[batch_indices]  # (batch, C, dim)
                batch_flat = batch_embs.reshape(-1, dim)  # (batch*C, dim)
                dot_prods = batch_flat @ query_emb.T  # (batch*C, num_tokens)
                dot_prods = dot_prods.reshape(len(batch_indices), C, -1)  # (batch, C, num_tokens)
                scores[start_idx:end_idx] = np.sum(np.max(dot_prods, axis=1), axis=1)
            
            # Step 3: Get top-k by exact MaxSim score
            if len(scores) > k:
                top_indices = np.argpartition(scores, -k)[-k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]
            
            result_doc_indices = candidate_doc_indices[top_indices]
            result_scores = scores[top_indices]
            
            all_doc_ids.append([self.doc_ids[idx] for idx in result_doc_indices])
            all_scores.append([float(s) for s in result_scores])
        
        return all_doc_ids, all_scores
    
    def save(self, output_path: str):
        """Save index to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving FAISS index to {output_path}...")
        
        # Save FAISS index separately
        faiss_path = output_path.with_suffix('.faiss')
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, str(faiss_path))
        else:
            faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save metadata
        with open(output_path, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'embeddings': self.embeddings,
                'use_gpu': self.use_gpu
            }, f)
        
        print(f"Index saved: {len(self.doc_ids)} documents")
    
    @classmethod
    def load(cls, input_path: str):
        """Load index from disk."""
        input_path = Path(input_path)
        
        print(f"Loading FAISS index from {input_path}...")
        
        # Load metadata
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls(use_gpu=data['use_gpu'])
        index.doc_ids = data['doc_ids']
        index.embeddings = data['embeddings']
        
        # Load FAISS index
        faiss_path = input_path.with_suffix('.faiss')
        index.faiss_index = faiss.read_index(str(faiss_path))
        
        if index.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index.faiss_index = faiss.index_cpu_to_gpu(res, 0, index.faiss_index)
        
        print(f"Index loaded: {len(index.doc_ids)} documents")
        return index
    
    def get_index_size_mb(self) -> float:
        """Get index size in MB (embeddings only)."""
        if self.embeddings is None:
            return 0.0
        return self.embeddings.nbytes / (1024 * 1024)
