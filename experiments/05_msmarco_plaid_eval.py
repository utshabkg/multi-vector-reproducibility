#!/usr/bin/env python3
"""
Evaluate MS-MARCO using ColBERT's IndexScorer with PLAID index and ConstBERT embeddings.

This script properly reproduces the paper's setup by:
1. Using ColBERT's IndexScorer (same infrastructure as authors)
2. Using PLAID index format with IVF (same as authors)
3. Using ConstBERT for query encoding
4. Using ranker.rank() for PLAID search

This is the CORRECT way to evaluate - using ColBERT's search infrastructure.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

from colbert.infra import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from data.loaders import MSMARCODataLoader
from models.constbert_wrapper import ConstBERTWrapper
from evaluation.metrics import compute_mrr, compute_recall, compute_map


def main():
    parser = argparse.ArgumentParser(description='Evaluate MS-MARCO with PLAID using ColBERT infrastructure')
    parser.add_argument('--index_name', type=str,
                       default='constbert_plaid_index',
                       help='Name of the index (relative to index_root)')
    parser.add_argument('--index_root', type=str,
                       default='/media/12TB/shared/datasets/indices/msmarco-passage',
                       help='Root directory containing indexes')
    parser.add_argument('--k', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--batch_size', type=int, default=128, help='Query encoding batch size')
    parser.add_argument('--output', type=str, default='results/exp4_plaid_dev_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MS-MARCO Evaluation with PLAID (ColBERT IndexScorer)")
    print("="*80)
    print(f"Index: {args.index_root}/{args.index_name}")
    print(f"k: {args.k}")
    print()
    
    # Load data
    print("Loading MS-MARCO data...")
    loader = MSMARCODataLoader()
    queries_dict = loader.load_queries('dev')
    qrels = loader.load_qrels('dev')
    
    print(f"  - {len(queries_dict):,} queries")
    print(f"  - {len(qrels):,} qrels")
    
    # Load doc_ids from index (for converting PIDs to doc IDs)
    index_path = os.path.join(args.index_root, args.index_name)
    doc_ids = np.load(os.path.join(index_path, "doc_ids.npy"))
    print(f"  - {len(doc_ids):,} documents in index")
    
    # Initialize ConstBERT for query encoding
    print("\nInitializing ConstBERT...")
    constbert = ConstBERTWrapper(model_name="pinecone/ConstBERT", batch_size=args.batch_size)
    
    # Encode all queries with ConstBERT
    print(f"\nEncoding {len(queries_dict)} queries with ConstBERT...")
    query_ids = sorted(queries_dict.keys())
    query_texts = [queries_dict[qid] for qid in query_ids]
    
    Q_all = constbert.encode_queries(query_texts, batch_size=args.batch_size)
    Q_all = torch.from_numpy(Q_all).half()  # (num_queries, num_tokens, 128), float16
    print(f"  Query embeddings: {Q_all.shape}")
    
    # Initialize ColBERT's IndexScorer with PLAID index
    print(f"\nInitializing IndexScorer...")
    print(f"  Index path: {index_path}")
    
    use_gpu = torch.cuda.is_available()
    ranker = IndexScorer(index_path, use_gpu=use_gpu, load_index_with_mmap=False)
    
    print(f"✅ IndexScorer initialized")
    print(f"   GPU: {use_gpu}")
    
    # Configure search parameters
    config = ColBERTConfig()
    if args.k <= 10:
        config.ncells = 1
        config.centroid_score_threshold = 0.5
        config.ndocs = 256
    elif args.k <= 100:
        config.ncells = 2
        config.centroid_score_threshold = 0.45
        config.ndocs = 1024
    else:
        config.ncells = 4
        config.centroid_score_threshold = 0.4
        config.ndocs = max(args.k * 4, 4096)
    
    print(f"   Search config: ncells={config.ncells}, ndocs={config.ndocs}, threshold={config.centroid_score_threshold}")
    
    # Search with pre-computed ConstBERT query embeddings
    print(f"\nSearching with PLAID (k={args.k})...")
    results = {}
    all_scores = {}
    start_time = time.time()
    
    if use_gpu:
        Q_all = Q_all.cuda()
    
    for idx, qid in enumerate(tqdm(query_ids, desc="Searching")):
        # Get query embedding for this query
        Q = Q_all[idx:idx+1]  # (1, num_tokens, 128)
        
        # Use IndexScorer.rank() for PLAID search
        pids, scores = ranker.rank(config, Q)
        
        # Take top-k and convert to doc IDs
        top_pids = pids[:args.k].cpu().numpy() if hasattr(pids, 'cpu') else pids[:args.k]
        top_scores = scores[:args.k].cpu().tolist() if hasattr(scores, 'cpu') else list(scores[:args.k])
        
        # Convert passage indices to document IDs with scores (expected format for metrics)
        doc_list = [(str(doc_ids[pid]), score) for pid, score in zip(top_pids, top_scores)]
        results[qid] = doc_list
        all_scores[qid] = top_scores
    
    search_time = time.time() - start_time
    avg_time_ms = search_time / len(query_ids) * 1000
    
    print(f"✅ Search completed in {search_time:.2f}s ({avg_time_ms:.1f}ms per query)")
    
    # Evaluate
    print("\nComputing metrics...")
    metrics = {
        'mrr@10': compute_mrr(results, qrels, k=10),
        'recall@50': compute_recall(results, qrels, k=50),
        'recall@200': compute_recall(results, qrels, k=200),
        'recall@1000': compute_recall(results, qrels, k=1000),
        'map@1000': compute_map(results, qrels, k=1000),
    }
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"  ncells={config.ncells}, ndocs={config.ndocs}, threshold={config.centroid_score_threshold}")
    print(f"\n  MRR@10:      {metrics['mrr@10']:.4f}")
    print(f"  Recall@50:   {metrics['recall@50']:.4f}")
    print(f"  Recall@200:  {metrics['recall@200']:.4f}")
    print(f"  Recall@1000: {metrics['recall@1000']:.4f}")
    print(f"  MAP@1000:    {metrics['map@1000']:.4f}")
    print(f"\n  Avg latency: {avg_time_ms:.1f}ms per query")
    
    # Comparison with paper
    print("\n" + "-"*80)
    print("COMPARISON WITH PAPER:")
    print("-"*80)
    print(f"  Metric        Ours      Paper     Diff")
    print(f"  MRR@10        {metrics['mrr@10']*100:.2f}%     39.04%    {(metrics['mrr@10']*100 - 39.04):+.2f}%")
    print(f"  Recall@1000   {metrics['recall@1000']*100:.2f}%     96.34%    {(metrics['recall@1000']*100 - 96.34):+.2f}%")
    print("="*80)
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        'metrics': metrics,
        'config': {
            'index': args.index_name,
            'k': args.k,
            'ncells': config.ncells,
            'ndocs': config.ndocs,
            'centroid_score_threshold': config.centroid_score_threshold,
        },
        'timing': {
            'total_time_s': search_time,
            'avg_time_ms': avg_time_ms,
        },
        'paper_comparison': {
            'paper_mrr10': 0.3904,
            'paper_recall1000': 0.9634,
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
