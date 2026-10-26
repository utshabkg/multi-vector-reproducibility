#!/usr/bin/env python3
"""
Build PLAID index from pre-computed ConstBERT embeddings.

This script:
1. Loads our existing ConstBERT embeddings (N, 32, 128)
2. Flattens them to (N*32, 128) format expected by PLAID
3. Trains PLAID centroids using k-means
4. Compresses embeddings using PLAID's residual quantization
5. Saves PLAID index for retrieval

This allows us to use the paper's actual indexing method (PLAID) while
keeping our optimized ConstBERT encoding (already done).
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from colbert.infra import ColBERTConfig
from colbert.indexing.codecs.residual import ResidualCodec
from data.loaders import MSMARCODataLoader
from models.constbert_wrapper import ConstBERTWrapper


def load_constbert_embeddings(embeddings_path, metadata_path, max_docs=None):
    """Load pre-computed ConstBERT embeddings."""
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)  # (N, 32, 128) float16
    
    print(f"Loading metadata from {metadata_path}...")
    # Handle both .npz and .pkl formats
    if metadata_path.endswith('.npz'):
        metadata = np.load(metadata_path)
        doc_ids = metadata['doc_ids']
    elif metadata_path.endswith('.pkl'):
        import pickle
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        doc_ids = metadata.get('doc_ids', metadata.get('docids'))
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_path}")
    
    # Limit to max_docs if specified (for testing)
    if max_docs is not None and max_docs < len(doc_ids):
        print(f"  ⚠️  TEST MODE: Using only first {max_docs:,} documents (out of {len(doc_ids):,})")
        embeddings = embeddings[:max_docs]
        doc_ids = doc_ids[:max_docs]
    
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Num documents: {len(doc_ids)}")
    print(f"  - Dtype: {embeddings.dtype}")
    
    return embeddings, doc_ids


def flatten_constbert_embeddings(embeddings):
    """
    Flatten ConstBERT embeddings from (N, 32, 128) to (N*32, 128).
    
    Also create doclens array indicating how many vectors per document.
    For ConstBERT, this is constant: [32, 32, 32, ...] 
    """
    N, C, dim = embeddings.shape
    assert C == 32, f"Expected C=32, got {C}"
    assert dim == 128, f"Expected dim=128, got {dim}"
    
    print(f"\nFlattening embeddings...")
    print(f"  From: {embeddings.shape} (N, C, dim)")
    
    # Flatten to (N*C, dim)
    flattened = embeddings.reshape(N * C, dim)
    
    # Create doclens: [32, 32, ..., 32] repeated N times
    doclens = [C] * N
    
    print(f"  To: {flattened.shape} (N*C, dim)")
    print(f"  Document lengths: constant {C} vectors/doc")
    
    return flattened, doclens


def train_plaid_centroids(embeddings_flat, config, sample_size=10_000_000):
    """
    Train PLAID centroids using k-means clustering.
    
    Args:
        embeddings_flat: (N*C, 128) flattened embeddings
        config: ColBERTConfig with nbits, kmeans_niters, dim
        sample_size: Number of vectors to sample for training
    
    Returns:
        ResidualCodec initialized with trained centroids
    """
    print(f"\n{'='*80}")
    print("Training PLAID Centroids")
    print(f"{'='*80}")
    
    # Sample vectors for training
    num_total_vectors = len(embeddings_flat)
    num_sample = min(sample_size, num_total_vectors)
    
    print(f"Total vectors: {num_total_vectors:,}")
    print(f"Sample size: {num_sample:,}")
    
    if num_sample < num_total_vectors:
        indices = np.random.choice(num_total_vectors, size=num_sample, replace=False)
        sample = embeddings_flat[indices]
    else:
        sample = embeddings_flat
    
    # Convert to torch
    sample_torch = torch.from_numpy(sample).float()
    
    # Determine number of centroids (partitions)
    # ColBERT uses 2^(ceil(log2(sqrt(num_embeddings))))
    # For MS-MARCO with ~283M vectors, this is 2^14 = 16384
    num_partitions = int(2 ** np.ceil(np.log2(np.sqrt(num_total_vectors))))
    num_partitions = max(1024, min(num_partitions, 65536))  # Clamp between 1K-64K
    
    print(f"Number of centroids (partitions): {num_partitions:,}")
    print(f"K-means iterations: {config.kmeans_niters}")
    print(f"Embedding dim: {config.dim}")
    print(f"Quantization bits: {config.nbits}")
    
    # Run k-means
    import faiss
    
    print(f"\nRunning FAISS k-means...")
    start_time = time.time()
    
    kmeans = faiss.Kmeans(
        d=config.dim,
        k=num_partitions,
        niter=config.kmeans_niters,
        verbose=True,
        gpu=torch.cuda.is_available(),
        seed=123
    )
    
    kmeans.train(sample_torch.numpy())
    
    elapsed = time.time() - start_time
    print(f"✅ K-means completed in {elapsed:.1f}s")
    
    # Get centroids
    centroids = torch.from_numpy(kmeans.centroids)
    print(f"Centroids shape: {centroids.shape}")
    
    # Train residual codec (bucket quantization)
    print(f"\nTraining residual codec...")
    codec = train_residual_codec(sample_torch, centroids, config)
    
    return codec


def train_residual_codec(sample, centroids, config):
    """
    Train residual quantization buckets.
    
    This computes statistics on the residuals to determine
    optimal bucket boundaries for quantization.
    """
    from colbert.indexing.codecs.residual import ResidualCodec
    
    # Move to GPU if available
    if torch.cuda.is_available():
        sample = sample.cuda().half()
        centroids = centroids.cuda().half()
    
    # Assign each vector to nearest centroid
    print("  - Finding nearest centroids...")
    with torch.no_grad():
        # Compute distances in batches
        batch_size = 100_000
        assignments = []
        for i in range(0, len(sample), batch_size):
            batch = sample[i:i+batch_size]
            # Cosine similarity (embeddings are normalized)
            scores = batch @ centroids.T
            assignments.append(scores.argmax(dim=1))
        assignments = torch.cat(assignments)
    
    # Compute residuals in batches to save memory
    print("  - Computing residuals...")
    residuals_list = []
    batch_size = 100_000
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i+batch_size]
        batch_assignments = assignments[i:i+batch_size]
        residuals_batch = batch - centroids[batch_assignments]
        # Move to CPU immediately to free GPU memory
        residuals_list.append(residuals_batch.cpu())
        del residuals_batch
    residuals = torch.cat(residuals_list)
    del residuals_list
    
    # Compute bucket statistics on CPU
    print("  - Computing bucket boundaries...")
    num_options = 2 ** config.nbits
    
    # Sample residuals for quantile computation (to save memory)
    # Use 100K samples instead of all 10M (100K × 128 = 12.8M elements, manageable)
    num_residual_samples = min(100_000, len(residuals))
    if num_residual_samples < len(residuals):
        indices = torch.randperm(len(residuals))[:num_residual_samples]
        residuals_sample = residuals[indices]
    else:
        residuals_sample = residuals
    
    # Flatten residuals across all dimensions and convert to numpy for quantile
    residuals_flat = residuals_sample.flatten().numpy()
    
    # Compute quantiles using numpy (more memory-efficient)
    quantiles_np = np.linspace(0, 1, num_options + 1)[1:-1]  # Exclude 0 and 1
    bucket_cutoffs = torch.from_numpy(np.quantile(residuals_flat, quantiles_np)).float()
    
    # Create bucket weights (midpoints of buckets)
    bucket_weights = torch.zeros(num_options)
    bucket_weights[0] = float(residuals_flat.min())
    for i in range(1, num_options):
        bucket_weights[i] = (bucket_cutoffs[i-1] + bucket_cutoffs[i]) / 2 if i < num_options - 1 else float(residuals_flat.max())
    
    # Compute average residual (on sample to save memory)
    avg_residual = residuals_sample.mean()
    
    print(f"  - Bucket cutoffs: {bucket_cutoffs.numpy()[:5]}... ({len(bucket_cutoffs)} buckets)")
    print(f"  - Average residual: {avg_residual.item():.6f}")
    print(f"  - Used {num_residual_samples:,} residual samples for statistics")
    
    # Create codec (all tensors already on CPU)
    codec = ResidualCodec(
        config=config,
        centroids=centroids.cpu(),
        avg_residual=avg_residual,
        bucket_cutoffs=bucket_cutoffs,
        bucket_weights=bucket_weights
    )
    
    return codec


def build_plaid_index(embeddings_path, metadata_path, index_path, config, max_docs=None):
    """
    Build complete PLAID index from ConstBERT embeddings.
    
    Args:
        max_docs: If specified, only use first N documents (for testing)
    """
    print(f"\n{'='*80}")
    print("Building PLAID Index from ConstBERT Embeddings")
    print(f"{'='*80}\n")
    
    # Create index directory
    os.makedirs(index_path, exist_ok=True)
    
    # Load embeddings
    embeddings, doc_ids = load_constbert_embeddings(embeddings_path, metadata_path, max_docs=max_docs)
    
    # Flatten for PLAID
    embeddings_flat, doclens = flatten_constbert_embeddings(embeddings)
    
    # Train PLAID codec
    codec = train_plaid_centroids(embeddings_flat, config)
    
    # Save codec
    print(f"\nSaving codec to {index_path}...")
    codec.save(index_path)
    print("  ✅ Codec saved")
    
    # Compress and save embeddings in chunks
    print(f"\nCompressing and saving embeddings...")
    chunk_size = 50_000  # Documents per chunk
    num_chunks = int(np.ceil(len(doc_ids) / chunk_size))
    
    print(f"  - Total documents: {len(doc_ids):,}")
    print(f"  - Chunk size: {chunk_size:,} documents")
    print(f"  - Number of chunks: {num_chunks}")
    
    offset = 0
    for chunk_idx in tqdm(range(num_chunks), desc="Compressing chunks"):
        # Get chunk of documents
        start_doc = chunk_idx * chunk_size
        end_doc = min((chunk_idx + 1) * chunk_size, len(doc_ids))
        
        # Get corresponding vectors
        start_vec = start_doc * 32
        end_vec = end_doc * 32
        
        # Extract chunk embeddings
        chunk_embs = torch.from_numpy(embeddings_flat[start_vec:end_vec])
        chunk_doclens = doclens[start_doc:end_doc]
        
        # Compress with PLAID
        compressed = codec.compress(chunk_embs)
        
        # Save chunk files
        path_prefix = os.path.join(index_path, str(chunk_idx))
        compressed.save(path_prefix)
        
        # Save doclens
        import ujson
        doclens_path = os.path.join(index_path, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as f:
            ujson.dump(chunk_doclens, f)
        
        # Save metadata
        metadata_path_chunk = os.path.join(index_path, f'{chunk_idx}.metadata.json')
        with open(metadata_path_chunk, 'w') as f:
            metadata = {
                'passage_offset': offset,
                'num_passages': len(chunk_doclens),
                'num_embeddings': len(compressed)
            }
            ujson.dump(metadata, f)
        
        offset += len(chunk_doclens)
    
    # Save plan
    print(f"\nSaving index plan...")
    plan_path = os.path.join(index_path, 'plan.json')
    import ujson
    plan = {
        'num_chunks': num_chunks,
        'num_partitions': len(codec.centroids),
        'num_embeddings': len(embeddings_flat),
        'avg_doclen': 32,  # ConstBERT always has 32 vectors/doc
        'num_documents': len(doc_ids)
    }
    with open(plan_path, 'w') as f:
        ujson.dump(plan, f, indent=2)
    
    print(f"  ✅ Plan saved")
    
    # Save doc_ids mapping with index for self-contained retrieval
    print(f"\nSaving document ID mapping...")
    doc_ids_path = os.path.join(index_path, 'doc_ids.npy')
    np.save(doc_ids_path, doc_ids)
    print(f"  ✅ Document IDs saved ({len(doc_ids):,} documents)")
    
    # Calculate total index size
    import subprocess
    result = subprocess.run(['du', '-sh', index_path], capture_output=True, text=True)
    index_size = result.stdout.strip().split()[0]
    
    print(f"\n{'='*80}")
    print(f"✅ PLAID Index Built Successfully!")
    print(f"{'='*80}")
    print(f"Index path: {index_path}")
    print(f"Index size: {index_size}")
    print(f"Documents: {len(doc_ids):,}")
    print(f"Vectors: {len(embeddings_flat):,}")
    print(f"Centroids: {len(codec.centroids):,}")
    print(f"Quantization: {config.nbits}-bit")
    print(f"{'='*80}\n")


def main():
    """Build PLAID index for MS-MARCO."""
    
    parser = argparse.ArgumentParser(description='Build PLAID index from ConstBERT embeddings')
    parser.add_argument('--test', type=int, default=None, metavar='N',
                        help='Test mode: only use first N documents (e.g., --test 10000)')
    args = parser.parse_args()
    
    # Configuration
    config = ColBERTConfig(
        dim=128,
        nbits=1,  # PLAID 1-bit quantization (as in paper)
        kmeans_niters=4,
        doc_maxlen=220,
        index_bsize=64
    )
    
    # Paths - embeddings are temporary input, index is self-contained output
    embeddings_path = "indices/msmarco-passage/msmarco-passage-constbert/constbert_msmarco_embeddings.npy"
    metadata_path = "indices/msmarco-passage/msmarco-passage-constbert/constbert_msmarco_faiss_index.pkl"
    
    if args.test:
        index_path = "indices/msmarco-passage/constbert_plaid_index_test"
        print(f"🧪 TEST MODE: Building index with {args.test:,} documents")
    else:
        index_path = "indices/msmarco-passage/constbert_plaid_index"
    
    print(f"Input embeddings: {embeddings_path}")
    print(f"Input metadata: {metadata_path}")
    print(f"Output index: {index_path}")
    print(f"\nNote: Index will be self-contained with all necessary metadata.\n")
    
    # Build index
    build_plaid_index(embeddings_path, metadata_path, index_path, config, max_docs=args.test)


if __name__ == "__main__":
    main()
