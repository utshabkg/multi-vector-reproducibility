#!/usr/bin/env python3
"""
ColBERT Indexing Implementation
Encodes documents and builds FAISS index for efficient retrieval.
"""

import torch
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import pickle
import json
import logging

try:
    from .colbert_model import ColBERT
except ImportError:
    from colbert_model import ColBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColBERTIndexer:
    """
    Indexes documents using ColBERT model.
    Creates FAISS index for efficient MaxSim retrieval.
    """

    def __init__(
        self,
        model: ColBERT,
        index_path: str,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize indexer.

        Args:
            model: Trained ColBERT model
            index_path: Directory to save index
            batch_size: Batch size for encoding
            device: Device to use
        """
        self.model = model.to(device)
        self.model.eval()
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = device

        # Index components - MEMORY EFFICIENT VERSION
        # Instead of loading all embeddings in RAM, we store them on disk
        self.embeddings_dir = self.index_path / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        self.doc_ids = []         # List of document IDs
        self.doc_lengths = []     # List of document lengths (Nd)
        self.faiss_index = None   # FAISS index

        # Checkpoint file paths
        self.metadata_checkpoint = self.index_path / "metadata_checkpoint.json"

        logger.info(f"Indexer initialized: {index_path}")
        logger.info(
            f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        logger.info(f"Embedding dim: {model.embedding_dim}")

        # Try to load existing checkpoint (both metadata and FAISS index)
        if self.metadata_checkpoint.exists():
            logger.info(f"Found existing checkpoint, attempting to load...")
            try:
                self.load_checkpoint()
                logger.info(
                    f"✅ Loaded checkpoint: {len(self.doc_ids):,} documents, {sum(self.doc_lengths):,} embeddings")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.warning(f"Starting from scratch...")

    def index_documents(
        self,
        documents: List[Tuple[str, str]],  # [(doc_id, doc_text), ...]
        show_progress: bool = True,
        checkpoint_every: int = 100000,
        resume: bool = True
    ):
        """
        Index a list of documents with checkpointing support.

        Args:
            documents: List of (doc_id, doc_text) tuples
            show_progress: Show progress bar
            checkpoint_every: Save checkpoint every N documents
            resume: Resume from checkpoint if exists
        """
        logger.info(f"Indexing {len(documents):,} documents...")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Checkpoint every: {checkpoint_every:,} documents")

        # Check for existing checkpoint (using new filename to preserve old backup)
        checkpoint_file = self.index_path / "encoding_checkpoint_v2.pkl"
        # old_checkpoint_file = self.index_path / "encoding_checkpoint.pkl"
        start_idx = 0

        # Try to convert old checkpoint to new format if it exists
        # if resume and old_checkpoint_file.exists() and not checkpoint_file.exists():
        #     logger.info(f"🔄 Found old checkpoint format, converting to new disk-based format...")
        #     logger.info(f"   This will take a few minutes but saves your progress!")
        #     try:
        #         old_checkpoint = pickle.load(open(old_checkpoint_file, 'rb'))
        #         old_embeddings = old_checkpoint['doc_embeddings']
        #         self.doc_ids = old_checkpoint['doc_ids']
        #         self.doc_lengths = old_checkpoint['doc_lengths']
        #         start_idx = old_checkpoint['next_idx']

        #         logger.info(f"   Converting {len(old_embeddings):,} document embeddings to disk...")
        #         for i, emb in enumerate(tqdm(old_embeddings, desc="Converting embeddings")):
        #             emb_file = self.embeddings_dir / f"doc_{i:08d}.npy"
        #             if isinstance(emb, torch.Tensor):
        #                 np.save(emb_file, emb.cpu().numpy())
        #             else:
        #                 np.save(emb_file, emb)

        #         # Save new checkpoint format
        #         checkpoint_data = {
        #             'doc_ids': self.doc_ids,
        #             'doc_lengths': self.doc_lengths,
        #             'next_idx': start_idx
        #         }
        #         pickle.dump(checkpoint_data, open(checkpoint_file, 'wb'))
        #         logger.info(f"   ✅ Conversion complete! Resuming from document {start_idx:,}")
        #         del old_embeddings, old_checkpoint  # Free memory

        #     except Exception as e:
        #         logger.warning(f"   ⚠️ Conversion failed: {e}")
        #         logger.warning(f"   Starting from scratch...")
        #         start_idx = 0
        #         self.doc_ids = []
        #         self.doc_lengths = []

        # # Load existing new-format checkpoint if it exists
        # elif resume and checkpoint_file.exists():
        if resume and checkpoint_file.exists():
            logger.info(f"📦 Loading checkpoint from {checkpoint_file}")
            checkpoint_data = pickle.load(open(checkpoint_file, 'rb'))
            self.doc_ids = checkpoint_data['doc_ids']
            self.doc_lengths = checkpoint_data['doc_lengths']
            start_idx = checkpoint_data['next_idx']
            logger.info(f"   ✓ Resuming from document {start_idx:,}")

        # Process documents in batches
        num_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        start_batch = start_idx // self.batch_size

        with torch.no_grad():
            for i in tqdm(
                range(start_batch, num_batches),
                desc="Encoding documents",
                disable=not show_progress,
                initial=start_batch,
                total=num_batches
            ):
                batch_start_idx = i * self.batch_size
                batch_end_idx = min((i + 1) * self.batch_size, len(documents))

                batch_docs = documents[batch_start_idx:batch_end_idx]
                batch_ids = [doc_id for doc_id, _ in batch_docs]
                batch_texts = [text for _, text in batch_docs]

                # Encode documents: [batch_size, Nd, dim]
                doc_embeddings = self.model.doc(batch_texts)

                # Store embeddings ON DISK to save RAM
                for j, doc_emb in enumerate(doc_embeddings):
                    # doc_emb: [Nd, dim]
                    doc_idx = batch_start_idx + j
                    emb_file = self.embeddings_dir / f"doc_{doc_idx:08d}.npy"
                    np.save(emb_file, doc_emb.cpu().numpy())
                    self.doc_ids.append(batch_ids[j])
                    self.doc_lengths.append(doc_emb.size(0))

                # Save checkpoint periodically (only metadata, not embeddings)
                if (batch_end_idx % checkpoint_every) < self.batch_size and batch_end_idx < len(documents):
                    checkpoint_data = {
                        'doc_ids': self.doc_ids,
                        'doc_lengths': self.doc_lengths,
                        'next_idx': batch_end_idx
                    }
                    pickle.dump(checkpoint_data, open(checkpoint_file, 'wb'))
                    logger.info(
                        f"💾 Checkpoint saved at {batch_end_idx:,} documents")

        # Remove checkpoint file after successful completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("🗑️  Checkpoint file removed (indexing complete)")

        logger.info(f"✅ Encoded {len(self.doc_ids):,} documents")
        logger.info(f"   Total embeddings: {sum(self.doc_lengths):,}")
        logger.info(f"   Avg tokens/doc: {np.mean(self.doc_lengths):.1f}")

    def train_faiss_index(
        self,
        train_documents: List[Tuple[str, str]],
        nlist: int = 4096,
        m_pq: int = 64,
        nbits_pq: int = 8,
        show_progress: bool = True
    ):
        """
        Train FAISS index structure on sample documents.
        Must be called before add_documents_batch.

        Args:
            train_documents: List of (doc_id, doc_text) tuples for training
            nlist: Number of IVF clusters
            m_pq: PQ subquantizers
            nbits_pq: Bits per PQ code
            show_progress: Show progress
        """
        logger.info(
            f"Training FAISS index on {len(train_documents):,} documents...")

        # Encode training documents
        train_embeddings = []
        num_batches = (len(train_documents) +
                       self.batch_size - 1) // self.batch_size

        with torch.no_grad():
            for i in tqdm(
                range(num_batches),
                desc="Encoding training docs",
                disable=not show_progress
            ):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(train_documents))

                batch_texts = [text for _,
                               text in train_documents[start_idx:end_idx]]
                doc_embeddings = self.model.doc(batch_texts)

                for doc_emb in doc_embeddings:
                    train_embeddings.append(doc_emb.cpu().numpy())

        # Concatenate and sample 1M embeddings for training
        all_train_embs = np.vstack(train_embeddings)
        train_size = min(1000000, len(all_train_embs))

        if len(all_train_embs) > train_size:
            indices = np.random.choice(
                len(all_train_embs), train_size, replace=False)
            train_data = all_train_embs[indices]
        else:
            train_data = all_train_embs

        faiss.normalize_L2(train_data)

        # Create and train index
        dim = train_data.shape[1]
        quantizer = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m_pq, nbits_pq, faiss.METRIC_INNER_PRODUCT)

        logger.info(
            f"   Training IVF-PQ with {len(train_data):,} embeddings...")
        logger.info(f"   Index: IVF{nlist}_PQ{m_pq}x{nbits_pq}")
        logger.info(f"   This may take 10-30 minutes...")

        self.faiss_index.train(train_data)
        self.faiss_index.nprobe = 10

        logger.info(f"   ✅ Index trained!")
        del train_embeddings, all_train_embs, train_data

    def add_documents_batch(
        self,
        documents: List[Tuple[str, str]],
        show_progress: bool = False
    ):
        """
        Encode documents and add to pre-trained FAISS index.
        Memory-efficient: processes batch and immediately adds to index.
        Auto-saves checkpoint every 10K documents.

        Args:
            documents: List of (doc_id, doc_text) tuples
            show_progress: Show progress bar
        """
        if self.faiss_index is None:
            raise RuntimeError(
                "Must call train_faiss_index() before add_documents_batch()")

        batch_ids = [doc_id for doc_id, _ in documents]
        batch_texts = [text for _, text in documents]

        # Encode documents
        with torch.no_grad():
            doc_embeddings = self.model.doc(batch_texts)

        # Add to FAISS index immediately
        batch_embeddings = []
        for j, doc_emb in enumerate(doc_embeddings):
            emb_np = doc_emb.cpu().numpy()
            batch_embeddings.append(emb_np)

            # Track metadata
            self.doc_ids.append(batch_ids[j])
            self.doc_lengths.append(len(emb_np))

        # Add to index
        all_batch_embs = np.vstack(batch_embeddings)
        faiss.normalize_L2(all_batch_embs)
        self.faiss_index.add(all_batch_embs)

        del batch_embeddings, all_batch_embs

        # Auto-save checkpoint every 10K documents
        if len(self.doc_ids) % 10000 < self.batch_size:
            self.save_checkpoint()

    def save_checkpoint(self):
        """Save intermediate checkpoint including FAISS index and metadata"""
        try:
            # Save FAISS index to disk (critical for recovery!)
            if self.faiss_index is not None:
                # Check if it's a GPU index (compatible with different FAISS builds)
                if hasattr(self.faiss_index, 'getDevice'):
                    faiss_cpu = faiss.index_gpu_to_cpu(self.faiss_index)
                else:
                    faiss_cpu = self.faiss_index

                faiss.write_index(faiss_cpu, str(
                    self.index_path / "embeddings.faiss"))

            # Save metadata
            metadata = {
                'doc_ids': self.doc_ids,
                'doc_lengths': self.doc_lengths,
                'num_docs': len(self.doc_ids),
                'total_embeddings': sum(self.doc_lengths) if self.doc_lengths else 0
            }

            with open(self.metadata_checkpoint, 'w') as f:
                json.dump(metadata, f)

            logger.info(
                f"   💾 Checkpoint saved: {len(self.doc_ids):,} documents, {sum(self.doc_lengths):,} embeddings")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load checkpoint from disk including FAISS index and metadata"""
        # Load metadata
        with open(self.metadata_checkpoint, 'r') as f:
            metadata = json.load(f)

        self.doc_ids = metadata['doc_ids']
        self.doc_lengths = metadata['doc_lengths']

        # Load FAISS index if it exists
        faiss_path = self.index_path / "embeddings.faiss"
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))

            # Move to GPU if available and FAISS GPU is available
            if torch.cuda.is_available() and self.device == 'cuda':
                try:
                    import faiss.contrib.torch_utils
                    if hasattr(faiss, 'index_cpu_to_gpu'):
                        res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(
                            res, 0, self.faiss_index)
                        logger.info(f"   Moved FAISS index to GPU")
                except (ImportError, AttributeError):
                    logger.info(
                        f"   FAISS GPU not available, keeping index on CPU")

            logger.info(
                f"   Loaded FAISS index with {self.faiss_index.ntotal:,} vectors")

        logger.info(f"Checkpoint loaded: {len(self.doc_ids):,} documents")
        return set(self.doc_ids)  # Return set of already indexed doc IDs

    def build_faiss_index(
        self,
        use_gpu: bool = True,
        nlist: int = 2000,  # Number of clusters for IVF (paper default)
        m_pq: int = 16,  # PQ subquantizers (paper default)
        nbits_pq: int = 8,  # Bits per PQ code
        chunk_size: int = 100000  # Process embeddings in chunks to avoid OOM
    ):
        """
        Build memory-efficient FAISS index using IVF + Product Quantization.
        This compresses embeddings to use ~32x less memory while maintaining good accuracy.

        Args:
            use_gpu: Use GPU for FAISS (much faster)
            nlist: Number of IVF clusters (more = better accuracy, more memory)
            m_pq: Number of PQ subquantizers (must divide embedding_dim)
            nbits_pq: Bits per PQ code (8 = 256 centroids per subquantizer)
            chunk_size: Number of embeddings to process at once
        """
        logger.info(
            "Building memory-efficient FAISS index with Product Quantization...")

        # Get total number of embeddings and dimension from first doc file
        total_embeddings = sum(self.doc_lengths)
        first_emb_file = self.embeddings_dir / "doc_00000000.npy"
        dim = np.load(first_emb_file).shape[1]

        logger.info(f"   Total embeddings: {total_embeddings:,}")
        logger.info(f"   Embedding dim: {dim}")
        logger.info(f"   Index type: IVF{nlist}_PQ{m_pq}x{nbits_pq}")
        logger.info(
            f"   Memory reduction: ~{dim * 4 / m_pq:.1f}x (vs flat index)")
        logger.info(f"   Processing in chunks of {chunk_size:,}")

        # Create IVF + PQ index for memory efficiency
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(
            quantizer, dim, nlist, m_pq, nbits_pq, faiss.METRIC_INNER_PRODUCT)

        # Train the index on a sample by reading from disk
        logger.info(f"   Training IVF-PQ index...")
        train_size = min(1000000, total_embeddings)
        logger.info(f"   Sampling {train_size:,} embeddings for training...")

        # Sample embeddings for training by reading from disk
        train_embeddings = []
        embeddings_collected = 0
        sample_every = max(1, len(self.doc_ids) // train_size)

        for i in range(len(self.doc_ids)):
            if i % sample_every == 0 and embeddings_collected < train_size:
                emb_file = self.embeddings_dir / f"doc_{i:08d}.npy"
                doc_np = np.load(emb_file)
                # Sample from this document
                if len(doc_np) > 0:
                    step = max(1, len(doc_np) //
                               max(1, train_size // len(self.doc_ids)))
                    sampled = doc_np[::step]
                    train_embeddings.append(sampled)
                    embeddings_collected += len(sampled)

                    if embeddings_collected >= train_size:
                        break

        train_data = np.vstack(train_embeddings)[:train_size]
        faiss.normalize_L2(train_data)
        logger.info(f"   Training with {len(train_data):,} embeddings...")
        logger.info(f"   This may take 10-30 minutes...")
        index.train(train_data)
        logger.info(f"   ✅ Training complete!")
        del train_data, train_embeddings

        # Add embeddings in chunks by reading from disk
        logger.info("   Adding embeddings to index in chunks...")
        chunk_buffer = []
        chunk_size_current = 0
        processed = 0

        for i in tqdm(range(len(self.doc_ids)), desc="Adding to index"):
            # Load embedding from disk
            emb_file = self.embeddings_dir / f"doc_{i:08d}.npy"
            doc_np = np.load(emb_file)
            chunk_buffer.append(doc_np)
            chunk_size_current += len(doc_np)

            # Process chunk when it reaches target size
            if chunk_size_current >= chunk_size:
                chunk_data = np.vstack(chunk_buffer)
                faiss.normalize_L2(chunk_data)
                index.add(chunk_data)
                processed += len(chunk_data)

                # Clear buffer
                chunk_buffer = []
                chunk_size_current = 0

                if processed % 1000000 == 0:  # Log every 1M
                    logger.info(
                        f"   Added {processed:,} / {total_embeddings:,} embeddings")

        # Add remaining embeddings
        if chunk_buffer:
            chunk_data = np.vstack(chunk_buffer)
            faiss.normalize_L2(chunk_data)
            index.add(chunk_data)
            processed += len(chunk_data)
            logger.info(
                f"   Added {processed:,} / {total_embeddings:,} embeddings (final)")

        # Set search parameters
        # Search more clusters for better recall
        index.nprobe = min(32, nlist // 4)

        self.faiss_index = index
        logger.info(f"✅ FAISS index built successfully!")
        logger.info(f"   Index type: IVF{nlist}_PQ{m_pq}x{nbits_pq}")
        logger.info(f"   Total vectors: {index.ntotal:,}")
        logger.info(
            f"   Memory usage: ~{index.ntotal * m_pq / (1024**3):.2f} GB (compressed)")

        # Note: GPU acceleration for IVF-PQ can be added here if needed
        # but currently keeping on CPU for compatibility

        return index

    def save(self):
        """Save index to disk."""
        logger.info(f"Saving index to {self.index_path}...")

        # Save FAISS index
        if hasattr(self.faiss_index, 'getDevice'):
            # Move back to CPU for saving
            faiss_cpu = faiss.index_gpu_to_cpu(self.faiss_index)
        else:
            faiss_cpu = self.faiss_index

        faiss.write_index(faiss_cpu, str(self.index_path / "embeddings.faiss"))

        # Save metadata
        metadata = {
            'doc_ids': self.doc_ids,
            'doc_lengths': self.doc_lengths,
            'num_docs': len(self.doc_ids),
            'num_embeddings': sum(self.doc_lengths),
            'embedding_dim': self.model.embedding_dim
        }

        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        # Save config
        config = {
            'model_name': 'colbert',
            'embedding_dim': self.model.embedding_dim,
            'num_docs': len(self.doc_ids),
            'num_embeddings': sum(self.doc_lengths)
        }

        with open(self.index_path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

        logger.info("✅ Index saved successfully")
        logger.info(f"   Documents: {len(self.doc_ids):,}")
        logger.info(f"   Embeddings: {sum(self.doc_lengths):,}")
        logger.info(f"   Location: {self.index_path}")

    def load(self, use_gpu: bool = True):
        """Load index from disk."""
        logger.info(f"Loading index from {self.index_path}...")

        # Load FAISS index
        self.faiss_index = faiss.read_index(
            str(self.index_path / "embeddings.faiss"))

        if use_gpu and torch.cuda.is_available():
            try:
                if hasattr(faiss, 'StandardGpuResources'):
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(
                        res, 0, self.faiss_index)
                    logger.info("   Moved index to GPU")
            except (AttributeError, RuntimeError) as e:
                logger.info(f"   Keeping index on CPU: {e}")

        # Load metadata
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        self.doc_ids = metadata['doc_ids']
        self.doc_lengths = metadata['doc_lengths']

        logger.info("✅ Index loaded successfully")
        logger.info(f"   Documents: {len(self.doc_ids):,}")
        logger.info(f"   Embeddings: {sum(self.doc_lengths):,}")

    def get_document_embeddings(self, doc_idx: int) -> np.ndarray:
        """
        Get embeddings for a specific document.

        Args:
            doc_idx: Document index (0 to num_docs-1)

        Returns:
            embeddings: [Nd, dim] numpy array
        """
        # Calculate start and end indices in FAISS index
        start_idx = sum(self.doc_lengths[:doc_idx])
        end_idx = start_idx + self.doc_lengths[doc_idx]

        # Retrieve from FAISS
        if hasattr(self.faiss_index, 'getDevice'):
            # GPU index
            embeddings = self.faiss_index.reconstruct_n(
                start_idx, self.doc_lengths[doc_idx])
        else:
            # CPU index
            embeddings = np.stack([
                self.faiss_index.reconstruct(i)
                for i in range(start_idx, end_idx)
            ])

        return embeddings


if __name__ == "__main__":
    """Test indexing with dummy data."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    print("Testing ColBERT indexing...")

    # Create model
    model = ColBERT()
    print(
        f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy documents
    documents = [
        (f"doc_{i}", f"This is document number {i} with some sample text.")
        for i in range(100)
    ]

    # Create indexer
    indexer = ColBERTIndexer(
        model=model,
        index_path="indices/test_index",
        batch_size=10
    )

    # Index documents
    print("\n1. Indexing documents...")
    indexer.index_documents(documents)

    # Build FAISS index
    print("\n2. Building FAISS index...")
    indexer.build_faiss_index(use_gpu=torch.cuda.is_available())

    # Save index
    print("\n3. Saving index...")
    indexer.save()

    # Load index
    print("\n4. Testing index loading...")
    indexer2 = ColBERTIndexer(
        model=model,
        index_path="indices/test_index",
        batch_size=10
    )
    indexer2.load(use_gpu=torch.cuda.is_available())

    # Test retrieval of document embeddings
    print("\n5. Testing document embedding retrieval...")
    doc_emb = indexer2.get_document_embeddings(0)
    print(f"   Document 0 embeddings shape: {doc_emb.shape}")

    print("\n✅ Indexing test completed!")
