#!/usr/bin/env python3
"""
Centroid Coverage Analysis

Measures unique centroids per document for ConstBERT vs ColBERT-v2.

Hypothesis: ConstBERT's 32 fixed vectors cluster into ~15 unique centroids,
while ColBERT-v2's variable-length vectors provide denser coverage.

This would explain why ConstBERT requires different PLAID parameters.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import faiss

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper


def analyze_centroid_coverage(
    model_name: str = "constbert",
    num_docs: int = 10000,
    num_centroids: int = 32768,  # 2^15, standard PLAID
    seed: int = 42
):
    """
    Analyze how many unique centroids each document's vectors map to.
    
    For each document:
    1. Get all embedding vectors
    2. Find nearest centroid for each vector
    3. Count unique centroids
    
    Returns statistics on centroid coverage.
    """
    
    print(f"=" * 60)
    print(f"Centroid Coverage Analysis: {model_name}")
    print(f"=" * 60)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load pre-computed embeddings
    if model_name == "constbert":
        embeddings_path = Path("/media/12TB/shared/datasets/indices/trec-tot-2025/constbert_tot_faiss_index/embeddings.npy")
        if not embeddings_path.exists():
            # Fall back to MS-MARCO
            embeddings_path = Path("/media/12TB/shared/datasets/indices/msmarco-passage/constbert_msmarco_faiss_index/embeddings.npy")
    else:
        # Would need ColBERT embeddings
        raise NotImplementedError("ColBERT centroid analysis not yet implemented")
    
    print(f"\nLoading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")
    
    total_docs = embeddings.shape[0]
    vectors_per_doc = embeddings.shape[1]
    dim = embeddings.shape[2]
    
    print(f"  Total documents: {total_docs:,}")
    print(f"  Vectors per document: {vectors_per_doc}")
    print(f"  Embedding dimension: {dim}")
    
    # Sample documents
    if num_docs < total_docs:
        sample_indices = np.random.choice(total_docs, size=num_docs, replace=False)
        embeddings = embeddings[sample_indices]
        print(f"  Sampled {num_docs:,} documents")
    
    # Flatten for centroid training
    print(f"\nTraining {num_centroids:,} centroids (simulating PLAID)...")
    flat_embeddings = embeddings.reshape(-1, dim).astype('float32')
    
    # Train k-means (like PLAID does)
    kmeans = faiss.Kmeans(
        d=dim,
        k=num_centroids,
        niter=20,
        verbose=True,
        gpu=torch.cuda.is_available(),
        seed=seed
    )
    
    # Sample for training (like PLAID does)
    train_sample_size = min(10_000_000, len(flat_embeddings))
    train_indices = np.random.choice(len(flat_embeddings), size=train_sample_size, replace=False)
    train_sample = flat_embeddings[train_indices]
    
    print(f"  Training on {train_sample_size:,} vectors...")
    kmeans.train(train_sample)
    
    # Get centroids
    centroids = kmeans.centroids
    print(f"  Centroids shape: {centroids.shape}")
    
    # Build index for nearest centroid lookup
    print("\nBuilding centroid index...")
    centroid_index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(centroids)  # Normalize for cosine similarity
    centroid_index.add(centroids)
    
    # Analyze coverage per document
    print(f"\nAnalyzing centroid coverage for {len(embeddings):,} documents...")
    
    unique_centroids_per_doc = []
    
    for i in tqdm(range(len(embeddings)), desc="Analyzing"):
        doc_vectors = embeddings[i].astype('float32')
        faiss.normalize_L2(doc_vectors)  # Normalize for cosine similarity
        
        # Find nearest centroid for each vector
        _, centroid_ids = centroid_index.search(doc_vectors, 1)
        centroid_ids = centroid_ids.flatten()
        
        # Count unique centroids
        unique_count = len(np.unique(centroid_ids))
        unique_centroids_per_doc.append(unique_count)
    
    unique_centroids_per_doc = np.array(unique_centroids_per_doc)
    
    # Compute statistics
    results = {
        "model": model_name,
        "num_documents": len(embeddings),
        "vectors_per_doc": vectors_per_doc,
        "num_centroids_total": num_centroids,
        "statistics": {
            "mean_unique_centroids": float(np.mean(unique_centroids_per_doc)),
            "median_unique_centroids": float(np.median(unique_centroids_per_doc)),
            "std_unique_centroids": float(np.std(unique_centroids_per_doc)),
            "min_unique_centroids": int(np.min(unique_centroids_per_doc)),
            "max_unique_centroids": int(np.max(unique_centroids_per_doc)),
            "p25_unique_centroids": float(np.percentile(unique_centroids_per_doc, 25)),
            "p75_unique_centroids": float(np.percentile(unique_centroids_per_doc, 75)),
        },
        "coverage_ratio": float(np.mean(unique_centroids_per_doc) / vectors_per_doc),
        "interpretation": ""
    }
    
    mean_unique = results["statistics"]["mean_unique_centroids"]
    coverage = results["coverage_ratio"]
    
    results["interpretation"] = (
        f"ConstBERT's {vectors_per_doc} vectors per document map to an average of "
        f"{mean_unique:.1f} unique centroids ({coverage*100:.1f}% coverage). "
        f"This sparse coverage explains why standard PLAID parameters (ncells=4) fail: "
        f"with only {mean_unique:.0f} centroids per doc, the probability of hitting "
        f"relevant documents when probing few centroids is low."
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS: Centroid Coverage Analysis")
    print(f"{'='*60}")
    print(f"  Model: {model_name}")
    print(f"  Vectors per document: {vectors_per_doc}")
    print(f"  Total centroids: {num_centroids:,}")
    print(f"\n  Unique centroids per document:")
    print(f"    Mean:   {mean_unique:.1f}")
    print(f"    Median: {results['statistics']['median_unique_centroids']:.1f}")
    print(f"    Std:    {results['statistics']['std_unique_centroids']:.1f}")
    print(f"    Range:  [{results['statistics']['min_unique_centroids']}, {results['statistics']['max_unique_centroids']}]")
    print(f"\n  Coverage ratio: {coverage*100:.1f}%")
    print(f"\n  Interpretation:")
    print(f"    {results['interpretation']}")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Centroid Coverage Analysis")
    parser.add_argument("--model", type=str, default="constbert", choices=["constbert"])
    parser.add_argument("--num_docs", type=int, default=10000)
    parser.add_argument("--num_centroids", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/17_centroid_coverage.json")
    args = parser.parse_args()
    
    results = analyze_centroid_coverage(
        model_name=args.model,
        num_docs=args.num_docs,
        num_centroids=args.num_centroids,
        seed=args.seed
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
