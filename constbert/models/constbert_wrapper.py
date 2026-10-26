"""
Wrapper for ConstBERT model from HuggingFace.
Does not modify the original model code.
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoModel
from tqdm import tqdm
import pickle
from pathlib import Path


class ConstBERTWrapper:
    """
    Wrapper for ConstBERT model that provides encoding and retrieval functionality.
    """
    
    def __init__(
        self,
        model_name: str = "pinecone/ConstBERT",
        device: str = None,
        batch_size: int = 128,  # Increased default for 48GB VRAM
        use_compile: bool = True,  # torch.compile for speed
    ):
        """
        Initialize ConstBERT model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cuda', 'cpu', or None for auto)
            batch_size: Batch size for encoding
            use_compile: Whether to use torch.compile (PyTorch 2.0+)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Loading ConstBERT model from {model_name}...")
        print(f"Using device: {self.device}")
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if use_compile and self.device == "cuda":
            try:
                print("Compiling model with torch.compile for faster inference...")
                self.model = torch.compile(self.model, mode="max-autotune")
                print("Model compiled successfully!")
            except Exception as e:
                print(f"torch.compile not available or failed: {e}")
        
        self.batch_size = batch_size
        print(f"Model loaded successfully. Batch size: {batch_size}")
    
    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode queries into embeddings.
        
        Args:
            queries: List of query strings
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (num_queries, num_tokens, embedding_dim)
        """
        all_embeddings = []
        bs = batch_size if batch_size is not None else self.batch_size
        
        iterator = range(0, len(queries), bs)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding queries")
        
        with torch.no_grad():
            for i in iterator:
                batch = queries[i:i + bs]
                embeddings = self.model.encode_queries(batch)
                
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = None,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode documents into fixed-size embeddings.
        
        Args:
            documents: List of document strings
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array of shape (num_docs, C, embedding_dim)
            where C is the fixed number of vectors per document (e.g., 32)
        """
        all_embeddings = []
        bs = batch_size if batch_size is not None else self.batch_size
        
        iterator = range(0, len(documents), bs)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding documents")
        
        with torch.no_grad():
            for i in iterator:
                batch = documents[i:i + bs]
                embeddings = self.model.encode_documents(batch)
                
                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()
                
                all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    @staticmethod
    def max_sim(q_emb: np.ndarray, d_emb: np.ndarray) -> float:
        """
        Compute MaxSim score between query and document embeddings.
        
        Args:
            q_emb: Query embeddings of shape (num_query_tokens, embedding_dim)
            d_emb: Document embeddings of shape (C, embedding_dim)
        
        Returns:
            MaxSim score (float)
        """
        assert q_emb.ndim == 2 and d_emb.ndim == 2, \
            f"Expected 2D arrays, got shapes {q_emb.shape} and {d_emb.shape}"
        
        # Compute scores: (C, num_query_tokens) = (C, dim) @ (dim, num_query_tokens)
        scores = np.dot(d_emb, q_emb.T)
        
        # Max over document vectors (axis 0), then sum over query tokens (axis 0 after max)
        return float(np.sum(np.max(scores, axis=0)))
    
    def score_documents(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        doc_ids: List[str] = None,
        top_k: int = 1000
    ) -> List[Tuple[str, float]]:
        """
        Score documents against a query using MaxSim.
        
        Args:
            query_embedding: Query embedding (num_tokens, dim)
            document_embeddings: Document embeddings (num_docs, C, dim)
            doc_ids: List of document IDs (optional)
            top_k: Number of top documents to return
        
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        num_docs = len(document_embeddings)
        
        if doc_ids is None:
            doc_ids = [str(i) for i in range(num_docs)]
        
        scores = np.zeros(num_docs)
        
        # Compute MaxSim for each document
        for i in range(num_docs):
            scores[i] = self.max_sim(query_embedding, document_embeddings[i])
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [(doc_ids[i], scores[i]) for i in top_indices]
        return results
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: str):
        """Save embeddings to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving embeddings to {output_path}...")
        np.save(output_path, embeddings)
        print(f"Saved embeddings with shape {embeddings.shape}")
    
    def load_embeddings(self, input_path: str) -> np.ndarray:
        """Load embeddings from disk."""
        print(f"Loading embeddings from {input_path}...")
        embeddings = np.load(input_path)
        print(f"Loaded embeddings with shape {embeddings.shape}")
        return embeddings


class ConstBERTIndex:
    """
    Simple in-memory index for ConstBERT document embeddings.
    """
    
    def __init__(self):
        self.doc_ids = []
        self.embeddings = None
    
    def add_documents(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray
    ):
        """
        Add documents to the index.
        
        Args:
            doc_ids: List of document IDs
            embeddings: Document embeddings (num_docs, C, dim)
        """
        assert len(doc_ids) == len(embeddings), \
            f"Mismatch: {len(doc_ids)} doc_ids vs {len(embeddings)} embeddings"
        
        self.doc_ids.extend(doc_ids)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.concatenate([self.embeddings, embeddings], axis=0)
    
    def search(
        self,
        query_embeddings: np.ndarray,
        k: int = 1000,
        wrapper: ConstBERTWrapper = None
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for top-k documents for queries using vectorized MaxSim.
        
        Args:
            query_embeddings: Query embeddings (batch_size, num_tokens, dim) or (num_tokens, dim)
            k: Number of results to return per query
            wrapper: ConstBERTWrapper instance for scoring (unused, kept for compatibility)
        
        Returns:
            Tuple of (doc_ids, scores) where each is a list of lists
        """
        # Handle single query case
        if query_embeddings.ndim == 2:
            query_embeddings = query_embeddings[np.newaxis, :]  # (1, num_tokens, dim)
        
        batch_size = query_embeddings.shape[0]
        all_doc_ids = []
        all_scores = []
        
        for i in range(batch_size):
            query_emb = query_embeddings[i]  # (num_tokens, dim)
            
            # Efficient vectorized MaxSim:
            # Reshape embeddings: (num_docs, C, dim) -> (num_docs*C, dim)
            num_docs, C, dim = self.embeddings.shape
            reshaped_embs = self.embeddings.reshape(num_docs * C, dim)
            
            # Compute all dot products: (num_docs*C, num_tokens)
            dot_products = reshaped_embs @ query_emb.T
            
            # Reshape back and take max over C: (num_docs, C, num_tokens) -> (num_docs, num_tokens)
            dot_products = dot_products.reshape(num_docs, C, -1)
            max_sims = np.max(dot_products, axis=1)
            
            # Sum over query tokens: (num_docs,)
            scores = np.sum(max_sims, axis=1)
            
            # Get top-k efficiently
            if k < num_docs:
                top_indices = np.argpartition(scores, -k)[-k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]
            
            all_doc_ids.append([self.doc_ids[idx] for idx in top_indices])
            all_scores.append([float(scores[idx]) for idx in top_indices])
        
        return all_doc_ids, all_scores
    
    def save(self, output_path: str):
        """Save index to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving index to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump({
                'doc_ids': self.doc_ids,
                'embeddings': self.embeddings
            }, f)
        print(f"Index saved: {len(self.doc_ids)} documents")
    
    @classmethod
    def load(cls, input_path: str):
        """Load index from disk."""
        print(f"Loading index from {input_path}...")
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
        
        index = cls()
        index.doc_ids = data['doc_ids']
        index.embeddings = data['embeddings']
        
        print(f"Index loaded: {len(index.doc_ids)} documents")
        return index
    
    def get_index_size_mb(self) -> float:
        """Get index size in MB."""
        if self.embeddings is None:
            return 0.0
        return self.embeddings.nbytes / (1024 * 1024)


if __name__ == "__main__":
    # Test the wrapper
    print("Testing ConstBERT wrapper...")
    
    # Initialize model
    wrapper = ConstBERTWrapper(device="cpu", batch_size=2)
    
    # Test encoding
    queries = ["What is machine learning?", "Python programming tutorials"]
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning uses neural networks."
    ]
    
    print("\nEncoding queries...")
    q_embs = wrapper.encode_queries(queries, show_progress=False)
    print(f"Query embeddings shape: {q_embs.shape}")
    
    print("\nEncoding documents...")
    d_embs = wrapper.encode_documents(documents, show_progress=False)
    print(f"Document embeddings shape: {d_embs.shape}")
    
    print("\nComputing MaxSim scores...")
    for i, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for j, doc in enumerate(documents):
            score = wrapper.max_sim(q_embs[i], d_embs[j])
            print(f"  Doc {j}: {score:.4f} - {doc[:50]}...")
    
    print("\nWrapper test complete!")
