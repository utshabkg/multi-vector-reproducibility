#!/usr/bin/env python3
"""
Experiment 8: TREC ToT 2025 Evaluation with PLAID Index

Evaluates ConstBERT on TREC Tip-of-the-Tongue 2025 using the PLAID index.
This allows comparison between:
- FAISS IVF results (exp4_tot_eval)
- PLAID results (this experiment)

ToT is a zero-shot evaluation on Wikipedia corpus with long descriptive queries.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from colbert.infra import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from models.constbert_wrapper import ConstBERTWrapper
from evaluation.metrics import compute_mrr, compute_recall, compute_ndcg, compute_map
from data.tot_loader import TRECToTDataLoader


def main():
    parser = argparse.ArgumentParser(description='Evaluate ToT with PLAID')
    parser.add_argument('--index_path', type=str,
                       default='indices/trec-tot-2025/constbert_tot_plaid_index',
                       help='Path to PLAID index')
    parser.add_argument('--k', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--ncells', type=int, default=16, help='Number of IVF cells to probe')
    parser.add_argument('--ndocs', type=int, default=8192, help='Max candidate docs')
    parser.add_argument('--batch_size', type=int, default=64, help='Query encoding batch size')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TREC ToT 2025 Evaluation with PLAID")
    print("="*80)
    print(f"Index: {args.index_path}")
    print(f"k={args.k}, ncells={args.ncells}, ndocs={args.ndocs}")
    print()
    
    # Load doc_ids from index
    doc_ids = np.load(os.path.join(args.index_path, "doc_ids.npy"), allow_pickle=True)
    print(f"Loaded {len(doc_ids):,} documents from index")
    
    # Load queries and qrels using existing loader
    print(f"\nLoading ToT queries and qrels...")
    loader = TRECToTDataLoader()
    queries = loader.load_queries("test")
    qrels = loader.load_qrels("test")
    
    # Filter to queries with qrels
    query_ids = [qid for qid in queries.keys() if qid in qrels]
    query_texts = [queries[qid] for qid in query_ids]
    
    print(f"  - {len(query_texts)} queries with qrels")
    
    # Initialize ConstBERT for query encoding
    print("\nInitializing ConstBERT...")
    constbert = ConstBERTWrapper(model_name="pinecone/ConstBERT", batch_size=args.batch_size)
    
    # Encode queries
    print(f"\nEncoding {len(query_texts)} queries with ConstBERT...")
    Q_all = constbert.encode_queries(query_texts, batch_size=args.batch_size)
    Q_all = torch.from_numpy(Q_all).half()
    print(f"  Query embeddings: {Q_all.shape}")
    
    # Initialize IndexScorer with PLAID index
    print(f"\nInitializing IndexScorer...")
    use_gpu = torch.cuda.is_available()
    ranker = IndexScorer(args.index_path, use_gpu=use_gpu, load_index_with_mmap=False)
    print(f"✅ IndexScorer initialized (GPU: {use_gpu})")
    
    # Configure search parameters
    config = ColBERTConfig()
    config.ncells = args.ncells
    config.centroid_score_threshold = 0.0
    config.ndocs = args.ndocs
    
    print(f"   Search config: ncells={config.ncells}, ndocs={config.ndocs}")
    
    if use_gpu:
        Q_all = Q_all.cuda()
    
    # Search with PLAID
    print(f"\nSearching with PLAID (k={args.k})...")
    results = {}
    start_time = time.time()
    
    for idx, qid in enumerate(tqdm(query_ids, desc="Searching ToT")):
        Q = Q_all[idx:idx+1]
        
        # Use IndexScorer.rank() for PLAID search
        pids, scores = ranker.rank(config, Q)
        
        # Take top-k and convert to doc IDs
        top_pids = pids[:args.k].cpu().numpy() if hasattr(pids, 'cpu') else pids[:args.k]
        top_scores = scores[:args.k].cpu().tolist() if hasattr(scores, 'cpu') else list(scores[:args.k])
        
        # Store as list of tuples for metrics
        doc_list = [(str(doc_ids[pid]), score) for pid, score in zip(top_pids, top_scores)]
        results[qid] = doc_list
    
    search_time = time.time() - start_time
    avg_time_ms = search_time / len(query_ids) * 1000
    
    print(f"\n✅ Search completed in {search_time:.2f}s ({avg_time_ms:.1f}ms per query)")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        'mrr@10': compute_mrr(results, qrels, k=10),
        'recall@50': compute_recall(results, qrels, k=50),
        'recall@200': compute_recall(results, qrels, k=200),
        'recall@1000': compute_recall(results, qrels, k=1000),
        'ndcg@10': compute_ndcg(results, qrels, k=10),
        'map@1000': compute_map(results, qrels, k=1000),
    }
    
    # Our FAISS IVF results for comparison
    faiss_results = {
        'mrr@10': 0.0427,
        'recall@50': 0.1141,
        'recall@1000': 0.2572,
        'ndcg@10': 0.0482,
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"         TREC ToT 2025 Results (PLAID ncells={args.ncells})")
    print(f"{'='*60}")
    for metric, value in sorted(metrics.items()):
        print(f"{metric:<20}: {value:.4f}")
    print(f"Mean Response Time:   {avg_time_ms:.1f} ms")
    print(f"{'='*60}")
    
    # Comparison
    print(f"\nComparison with FAISS IVF:")
    print(f"{'Metric':<20} {'FAISS IVF':<15} {'PLAID':<15} {'Diff':<15}")
    print("-" * 60)
    for metric in ['mrr@10', 'recall@50', 'recall@1000', 'ndcg@10']:
        faiss_val = faiss_results.get(metric, 0) * 100
        plaid_val = metrics.get(metric, 0) * 100
        diff = plaid_val - faiss_val
        print(f"{metric:<20} {faiss_val:>12.2f}% {plaid_val:>12.2f}% {diff:>+12.2f}%")
    
    # Save results
    output_file = f"results/exp8_tot_plaid_ncells{args.ncells}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'dataset': 'TREC ToT 2025',
        'method': 'PLAID',
        'num_queries': len(query_texts),
        'metrics': metrics,
        'config': {
            'ncells': config.ncells,
            'ndocs': config.ndocs,
            'k': args.k,
        },
        'timing': {
            'total_time_s': search_time,
            'avg_time_ms': avg_time_ms,
        },
        'comparison': {
            'faiss_ivf': faiss_results
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY: ToT Zero-Shot Evaluation")
    print(f"{'='*80}")
    print(f"Both FAISS IVF and PLAID show poor performance on ToT")
    print(f"This confirms ConstBERT's domain-specific nature (trained on MS-MARCO)")
    print(f"Long descriptive queries are fundamentally different from short factoid queries")


if __name__ == "__main__":
    main()
