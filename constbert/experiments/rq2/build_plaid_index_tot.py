#!/usr/bin/env python3
"""
Build PLAID index for ToT corpus from pre-computed ConstBERT embeddings.

ToT corpus: 6.4M Wikipedia articles (vs MS-MARCO 8.8M passages)
Embeddings: Already exist at indices/trec-tot-2025/constbert_tot_faiss_index/embeddings.npy

This will enable comparison of FAISS IVF vs PLAID on ToT dataset.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from colbert.infra import ColBERTConfig
from colbert.indexing.codecs.residual import ResidualCodec


def load_tot_embeddings(embeddings_path, metadata_path):
    """Load pre-computed ConstBERT embeddings for ToT."""
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)  # (N, 32, 128) float16
    
    print(f"Loading metadata from {metadata_path}...")
    metadata = np.load(metadata_path, allow_pickle=True)
    doc_ids = metadata  # metadata.npy contains doc_ids directly
    
    print(f"  - Embeddings shape: {embeddings.shape}")
    print(f"  - Num documents: {len(doc_ids)}")
    print(f"  - Dtype: {embeddings.dtype}")
    
    return embeddings, doc_ids


def flatten_constbert_embeddings(embeddings):
    """Flatten ConstBERT embeddings from (N, 32, 128) to (N*32, 128)."""
    N, C, dim = embeddings.shape
    assert C == 32, f"Expected C=32, got {C}"
    assert dim == 128, f"Expected dim=128, got {dim}"
    
    print(f"\nFlattening embeddings...")
    print(f"  From: {embeddings.shape} (N, C, dim)")
    
    flattened = embeddings.reshape(N * C, dim)
    doclens = [C] * N
    
    print(f"  To: {flattened.shape} (N*C, dim)")
    print(f"  Document lengths: constant {C} vectors/doc")
    
    return flattened, doclens


def train_plaid_codec(embeddings_flat, num_partitions, config, sample_size=8_000_000):
    """Train PLAID centroids and residual codec using k-means."""
    import faiss
    
    print(f"\n{'='*80}")
    print("Training PLAID Codec")
    print(f"{'='*80}")
    
    num_total_vectors = len(embeddings_flat)
    num_sample = min(sample_size, num_total_vectors)
    
    print(f"Total vectors: {num_total_vectors:,}")
    print(f"Sample size: {num_sample:,}")
    print(f"Number of centroids: {num_partitions:,}")
    print(f"K-means iterations: {config.kmeans_niters}")
    
    # Sample vectors for training
    if num_sample < num_total_vectors:
        indices = np.random.choice(num_total_vectors, size=num_sample, replace=False)
        sample = embeddings_flat[indices].astype(np.float32)
    else:
        sample = embeddings_flat.astype(np.float32)
    
    # Run k-means
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
    
    kmeans.train(sample)
    elapsed = time.time() - start_time
    print(f"✅ K-means completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    centroids = torch.from_numpy(kmeans.centroids)
    print(f"Centroids shape: {centroids.shape}")
    
    # Train residual codec with proper bucket computation
    print(f"\nTraining residual codec...")
    sample_torch = torch.from_numpy(sample)
    
    if torch.cuda.is_available():
        sample_torch = sample_torch.cuda().half()
        centroids = centroids.cuda().half()
    
    # Assign to nearest centroids
    print("  - Finding nearest centroids...")
    batch_size = 100_000
    assignments = []
    for i in range(0, len(sample_torch), batch_size):
        batch = sample_torch[i:i+batch_size]
        scores = batch @ centroids.T
        assignments.append(scores.argmax(dim=1))
    assignments = torch.cat(assignments)
    
    # Compute residuals in batches to save memory
    print("  - Computing residuals...")
    residuals_list = []
    batch_size = 100_000
    for i in range(0, len(sample_torch), batch_size):
        batch = sample_torch[i:i+batch_size]
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
        if i < num_options - 1:
            bucket_weights[i] = (bucket_cutoffs[i-1] + bucket_cutoffs[i]) / 2
        else:
            bucket_weights[i] = float(residuals_flat.max())
    
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


def compress_and_save_chunks(embeddings_flat, doclens, codec, output_path, chunk_size=50_000, test_mode=False):
    """Compress embeddings in chunks and save PLAID format."""
    print(f"\n{'='*80}")
    print("Compressing and Saving Embeddings")
    print(f"{'='*80}")
    
    num_docs = len(doclens)
    num_chunks = (num_docs + chunk_size - 1) // chunk_size
    
    if test_mode:
        num_chunks = min(2, num_chunks)  # Test with just 2 chunks
        print(f"⚠️  TEST MODE: Using only {num_chunks} chunks")
    
    print(f"  - Total documents: {num_docs:,}")
    print(f"  - Vectors per doc: {doclens[0]}")
    print(f"  - Chunk size: {chunk_size:,} documents")
    print(f"  - Number of chunks: {num_chunks}")
    
    os.makedirs(output_path, exist_ok=True)
    
    # Move codec to GPU if available
    if torch.cuda.is_available():
        codec.centroids = codec.centroids.cuda().half()
    
    vecs_per_doc = doclens[0]  # Constant 32 for ConstBERT
    
    for chunk_idx in tqdm(range(num_chunks), desc="Compressing chunks"):
        start_doc = chunk_idx * chunk_size
        end_doc = min((chunk_idx + 1) * chunk_size, num_docs)
        
        start_vec = start_doc * vecs_per_doc
        end_vec = end_doc * vecs_per_doc
        
        chunk_emb = embeddings_flat[start_vec:end_vec]
        chunk_emb = torch.from_numpy(chunk_emb).float()
        
        if torch.cuda.is_available():
            chunk_emb = chunk_emb.cuda().half()
        
        # Compress with codec - returns ResidualEmbeddings object
        compressed = codec.compress(chunk_emb)
        
        # Save using the object's save method
        path_prefix = os.path.join(output_path, str(chunk_idx))
        compressed.save(path_prefix)
        
        # Save doclens for this chunk
        chunk_doclens = doclens[start_doc:end_doc]
        doclens_path = os.path.join(output_path, f'doclens.{chunk_idx}.json')
        import ujson
        with open(doclens_path, 'w') as f:
            ujson.dump(chunk_doclens, f)
    
    print(f"✅ Saved {num_chunks} chunks")


def build_ivf(output_path, num_partitions, num_chunks):
    """Build IVF (Inverted File) from compressed codes."""
    from collections import defaultdict
    
    print(f"\n{'='*80}")
    print("Building IVF (Inverted File)")
    print(f"{'='*80}")
    
    ivf_dict = defaultdict(list)
    global_offset = 0
    
    for chunk_idx in tqdm(range(num_chunks), desc="Building IVF"):
        codes_path = os.path.join(output_path, f"{chunk_idx}.codes.pt")
        codes = torch.load(codes_path, map_location='cpu')
        
        for local_idx, centroid_id in enumerate(codes):
            global_idx = global_offset + local_idx
            ivf_dict[centroid_id.item()].append(global_idx)
        
        global_offset += len(codes)
    
    print(f"Total embeddings indexed: {global_offset:,}")
    
    # Convert to ColBERT format
    ivf_list = []
    ivf_lengths = []
    
    for centroid_id in range(num_partitions):
        eids = ivf_dict.get(centroid_id, [])
        ivf_list.extend(eids)
        ivf_lengths.append(len(eids))
    
    ivf_tensor = torch.tensor(ivf_list, dtype=torch.int32)
    ivf_lengths_tensor = torch.tensor(ivf_lengths, dtype=torch.int32)
    
    non_empty = sum(1 for l in ivf_lengths if l > 0)
    print(f"IVF shape: {ivf_tensor.shape}")
    print(f"Non-empty partitions: {non_empty:,} / {num_partitions:,}")
    
    ivf_path = os.path.join(output_path, "ivf.pt")
    torch.save((ivf_tensor, ivf_lengths_tensor), ivf_path)
    
    print(f"✅ IVF saved to {ivf_path}")


def main():
    parser = argparse.ArgumentParser(description='Build PLAID index for ToT corpus')
    parser.add_argument('--embeddings_path', type=str,
                       default='indices/trec-tot-2025/constbert_tot_faiss_index/embeddings.npy',
                       help='Path to pre-computed embeddings')
    parser.add_argument('--metadata_path', type=str,
                       default='indices/trec-tot-2025/constbert_tot_faiss_index/metadata.npy',
                       help='Path to metadata (doc IDs)')
    parser.add_argument('--output_path', type=str,
                       default='indices/trec-tot-2025/constbert_tot_plaid_index',
                       help='Output PLAID index path')
    parser.add_argument('--num_partitions', type=int, default=8192,
                       help='Number of centroids (default: 8192 for ~200M vectors)')
    parser.add_argument('--kmeans_iters', type=int, default=4,
                       help='K-means iterations')
    parser.add_argument('--chunk_size', type=int, default=25000,
                       help='Documents per chunk (smaller for faster processing)')
    parser.add_argument('--sample_size', type=int, default=6_000_000,
                       help='Sample size for k-means training')
    parser.add_argument('--test_mode', action='store_true',
                       help='Test mode: use small data for quick testing')
    
    args = parser.parse_args()
    
    print("="*80)
    print("Building PLAID Index for TREC ToT 2025")
    print("="*80)
    print(f"Input embeddings: {args.embeddings_path}")
    print(f"Input metadata: {args.metadata_path}")
    print(f"Output index: {args.output_path}")
    print(f"Partitions (centroids): {args.num_partitions}")
    print()
    
    # Load embeddings
    embeddings, doc_ids = load_tot_embeddings(args.embeddings_path, args.metadata_path)
    
    # Flatten to (N*32, 128)
    embeddings_flat, doclens = flatten_constbert_embeddings(embeddings)
    
    # Configure PLAID
    config = ColBERTConfig()
    config.dim = 128
    config.nbits = 1  # 1-bit residual quantization
    config.kmeans_niters = args.kmeans_iters
    
    # Train codec
    codec = train_plaid_codec(embeddings_flat, args.num_partitions, config, args.sample_size)
    
    # Save codec
    os.makedirs(args.output_path, exist_ok=True)
    
    centroids_path = os.path.join(args.output_path, "centroids.pt")
    torch.save(codec.centroids, centroids_path)
    print(f"✅ Saved centroids to {centroids_path}")
    
    codec_path = os.path.join(args.output_path, "codec.pt")
    torch.save(codec, codec_path)
    print(f"✅ Saved codec to {codec_path}")
    
    # Save doclens
    doclens_path = os.path.join(args.output_path, "doclens.0.json")
    with open(doclens_path, 'w') as f:
        json.dump(doclens, f)
    print(f"✅ Saved doclens to {doclens_path}")
    
    # Save doc_ids
    docids_path = os.path.join(args.output_path, "doc_ids.npy")
    np.save(docids_path, doc_ids)
    print(f"✅ Saved doc_ids to {docids_path}")
    
    # Compress and save chunks
    num_docs = len(doclens)
    num_chunks = (num_docs + args.chunk_size - 1) // args.chunk_size
    
    if args.test_mode:
        num_chunks = min(2, num_chunks)  # Test with just 2 chunks
    
    compress_and_save_chunks(embeddings_flat, doclens, codec, args.output_path, args.chunk_size, args.test_mode)
    
    # Save plan
    plan = {
        'num_partitions': args.num_partitions,
        'num_embeddings': len(embeddings_flat),
        'num_docs': num_docs,
        'num_chunks': num_chunks,
        'avg_doclen': 32,
        'dim': 128,
        'nbits': 1
    }
    plan_path = os.path.join(args.output_path, "plan.json")
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"✅ Saved plan to {plan_path}")
    
    # Build IVF
    build_ivf(args.output_path, args.num_partitions, num_chunks)
    
    # Calculate storage
    total_size = sum(
        os.path.getsize(os.path.join(args.output_path, f))
        for f in os.listdir(args.output_path)
        if os.path.isfile(os.path.join(args.output_path, f))
    )
    print(f"\n{'='*80}")
    print("✅ PLAID Index Built Successfully!")
    print(f"{'='*80}")
    print(f"Index path: {args.output_path}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Documents: {num_docs:,}")
    print(f"Embeddings: {len(embeddings_flat):,}")


if __name__ == "__main__":
    main()
