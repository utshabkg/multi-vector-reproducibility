#!/usr/bin/env python3
"""
Experiment 7: TREC Deep Learning Track 2019/2020 Evaluation with PLAID Index

Evaluates ConstBERT on TREC DL 2019 and 2020 using the PLAID index infrastructure.
This allows comparison between:
- FAISS IVF results (exp2)
- PLAID results (this experiment)
- Paper reported results

Expected: PLAID may show similar ~8% degradation vs exact MaxSim as seen on MS-MARCO,
due to ConstBERT's fixed 32-vector representation being incompatible with PLAID assumptions.
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
from evaluation.metrics import evaluate_retrieval


def load_trec_queries(year):
    """Load TREC DL queries."""
    data_dir = Path("data/msmarco-passage")
    query_path = data_dir / f"msmarco-test{year}-queries.tsv"
    
    queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                query = parts[1]
                queries[qid] = query
    
    return queries


def load_trec_qrels(year):
    """Load TREC DL qrels (graded relevance 0-3)."""
    data_dir = Path("data/msmarco-passage")
    qrel_path = data_dir / f"{year}qrels-pass.txt"
    
    qrels = {}
    with open(qrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = relevance
    
    return qrels


def main():
    parser = argparse.ArgumentParser(description='Evaluate TREC DL with PLAID')
    parser.add_argument('--index_name', type=str,
                       default='constbert_plaid_index',
                       help='Name of the index')
    parser.add_argument('--index_root', type=str,
                       default='indices/msmarco-passage',
                       help='Root directory containing indexes')
    parser.add_argument('--k', type=int, default=1000, help='Number of results per query')
    parser.add_argument('--ncells', type=int, default=16, help='Number of IVF cells to probe')
    parser.add_argument('--ndocs', type=int, default=8192, help='Max candidate docs')
    parser.add_argument('--batch_size', type=int, default=128, help='Query encoding batch size')
    parser.add_argument('--year', type=int, choices=[2019, 2020], default=None,
                       help='Specific year to evaluate (default: both)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("TREC Deep Learning Track Evaluation with PLAID")
    print("="*80)
    print(f"Index: {args.index_root}/{args.index_name}")
    print(f"k={args.k}, ncells={args.ncells}, ndocs={args.ndocs}")
    print()
    
    # Load doc_ids from index
    index_path = os.path.join(args.index_root, args.index_name)
    doc_ids = np.load(os.path.join(index_path, "doc_ids.npy"))
    print(f"Loaded {len(doc_ids):,} documents from index")
    
    # Initialize ConstBERT for query encoding
    print("\nInitializing ConstBERT...")
    constbert = ConstBERTWrapper(model_name="pinecone/ConstBERT", batch_size=args.batch_size)
    
    # Initialize ColBERT's IndexScorer with PLAID index
    print(f"\nInitializing IndexScorer...")
    use_gpu = torch.cuda.is_available()
    ranker = IndexScorer(index_path, use_gpu=use_gpu, load_index_with_mmap=False)
    print(f"✅ IndexScorer initialized (GPU: {use_gpu})")
    
    # Configure search parameters - use larger ncells for better recall
    config = ColBERTConfig()
    config.ncells = args.ncells
    config.centroid_score_threshold = 0.0  # Disable threshold filtering
    config.ndocs = args.ndocs
    
    print(f"   Search config: ncells={config.ncells}, ndocs={config.ndocs}")
    
    # Paper results for comparison
    paper_results = {
        2019: {"ndcg@10": 0.7314},
        2020: {"ndcg@10": 0.7329}
    }
    
    # Our FAISS IVF results from exp2 for comparison
    faiss_results = {
        2019: {"ndcg@10": 0.6829},
        2020: {"ndcg@10": 0.6930}
    }
    
    # Determine which years to evaluate
    years = [args.year] if args.year else [2019, 2020]
    
    all_results = {}
    
    for year in years:
        print(f"\n{'='*80}")
        print(f"TREC DL {year} Evaluation")
        print(f"{'='*80}")
        
        # Load queries and qrels
        print(f"\nLoading TREC DL {year} data...")
        queries = load_trec_queries(year)
        qrels = load_trec_qrels(year)
        
        # Filter to queries with qrels
        query_ids = [qid for qid in queries.keys() if qid in qrels]
        query_texts = [queries[qid] for qid in query_ids]
        
        print(f"  - {len(query_texts)} queries with qrels")
        
        # Encode queries with ConstBERT
        print(f"\nEncoding {len(query_texts)} queries with ConstBERT...")
        Q_all = constbert.encode_queries(query_texts, batch_size=args.batch_size)
        Q_all = torch.from_numpy(Q_all).half()  # (num_queries, num_tokens, 128)
        print(f"  Query embeddings: {Q_all.shape}")
        
        if use_gpu:
            Q_all = Q_all.cuda()
        
        # Search with PLAID
        print(f"\nSearching with PLAID (k={args.k})...")
        results = {}
        start_time = time.time()
        
        for idx, qid in enumerate(tqdm(query_ids, desc=f"Searching TREC DL {year}")):
            # Get query embedding
            Q = Q_all[idx:idx+1]  # (1, num_tokens, 128)
            
            # Use IndexScorer.rank() for PLAID search
            pids, scores = ranker.rank(config, Q)
            
            # Take top-k and convert to doc IDs
            top_pids = pids[:args.k].cpu().numpy() if hasattr(pids, 'cpu') else pids[:args.k]
            top_scores = scores[:args.k].cpu().tolist() if hasattr(scores, 'cpu') else list(scores[:args.k])
            
            # Store as dict format for evaluation
            results[qid] = {str(doc_ids[pid]): score for pid, score in zip(top_pids, top_scores)}
        
        search_time = time.time() - start_time
        avg_time_ms = search_time / len(query_ids) * 1000
        
        print(f"\n✅ Search completed in {search_time:.2f}s ({avg_time_ms:.1f}ms per query)")
        
        # Evaluate with TREC metrics
        print(f"\nComputing evaluation metrics...")
        metrics_to_compute = ['ndcg@10', 'map@1000', 'recall@100', 'recall@1000', 'p@10']
        metrics = evaluate_retrieval(results, qrels, metrics=metrics_to_compute)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"              TREC DL {year} Results (PLAID)")
        print(f"{'='*60}")
        for metric, value in sorted(metrics.items()):
            print(f"{metric:<20}: {value:.4f}")
        print(f"Mean Response Time:   {avg_time_ms:.1f} ms")
        print(f"{'='*60}")
        
        # Comparison with paper and FAISS
        print(f"\nComparison:")
        print(f"{'Method':<20} {'NDCG@10':<15} {'vs Paper':<15}")
        print("-" * 50)
        
        our_ndcg = metrics['ndcg@10'] * 100
        paper_ndcg = paper_results[year]['ndcg@10'] * 100
        faiss_ndcg = faiss_results[year]['ndcg@10'] * 100
        
        print(f"{'Paper (ConstBERT32)':<20} {paper_ndcg:>12.2f}% {'(reference)':<15}")
        print(f"{'Our FAISS IVF (exp2)':<20} {faiss_ndcg:>12.2f}% {faiss_ndcg - paper_ndcg:>+12.2f}%")
        print(f"{'Our PLAID (ncells={})'.format(args.ncells):<20} {our_ndcg:>12.2f}% {our_ndcg - paper_ndcg:>+12.2f}%")
        
        # Store results
        all_results[year] = {
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
            'comparisons': {
                'paper_ndcg10': paper_results[year]['ndcg@10'],
                'faiss_ndcg10': faiss_results[year]['ndcg@10'],
                'plaid_ndcg10': metrics['ndcg@10'],
            }
        }
    
    # Save results
    output_file = f"results/exp7_trec_dl_plaid_ncells{args.ncells}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY: PLAID vs FAISS IVF vs Paper")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'Paper':<12} {'FAISS IVF':<12} {'PLAID':<12} {'PLAID-Paper':<12}")
    print("-" * 65)
    
    for year in years:
        paper = paper_results[year]['ndcg@10'] * 100
        faiss = faiss_results[year]['ndcg@10'] * 100
        plaid = all_results[year]['metrics']['ndcg@10'] * 100
        diff = plaid - paper
        print(f"{'TREC DL ' + str(year):<15} {paper:>10.2f}% {faiss:>10.2f}% {plaid:>10.2f}% {diff:>+10.2f}%")
    
    print(f"{'='*80}")
    print("\nNote: PLAID degradation is expected due to ConstBERT's fixed 32-vector")
    print("representation being incompatible with PLAID's token-based assumptions.")
    print("See ISSUE_RETRIEVAL_STUCK.md Issue 3 for detailed analysis.")


if __name__ == "__main__":
    main()
