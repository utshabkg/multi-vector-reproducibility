#!/usr/bin/env python3
"""
PLAID Parameter Sweep for ConstBERT

Systematic exploration of PLAID parameters to document the reproducibility gap.
Tests ncells ∈ [4, 8, 16, 32, 64] and threshold ∈ [0.3, 0.4, 0.5]

Per supervisor feedback: "Shows you exhaustively searched for the missing parameters"
"""

import os
import sys
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "external" / "ColBERT"))

from colbert.infra import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from models.constbert_wrapper import ConstBERTWrapper


@dataclass
class PLAIDConfig:
    ncells: int
    centroid_score_threshold: float
    ndocs: int = 8192


def run_plaid_search(
    ranker: IndexScorer,
    Q_all: torch.Tensor,
    query_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    doc_ids: np.ndarray,
    ncells: int,
    threshold: float,
    ndocs: int = 8192,
    k: int = 10
) -> Dict:
    """Run PLAID search with given config and compute MRR@10."""
    
    print(f"\n  Testing ncells={ncells}, threshold={threshold}...")
    
    # Configure search
    config = ColBERTConfig()
    config.ncells = ncells
    config.centroid_score_threshold = threshold
    config.ndocs = ndocs
    
    # Run search
    start_time = time.time()
    
    mrr_sum = 0.0
    num_queries = 0
    
    for idx, qid in enumerate(query_ids):
        if qid not in qrels:
            continue
            
        Q = Q_all[idx:idx+1]  # (1, num_tokens, 128)
        pids, scores = ranker.rank(config, Q)
        
        # Take top-k and convert to doc IDs
        top_pids = pids[:k].cpu().numpy() if hasattr(pids, 'cpu') else pids[:k]
        
        # Compute reciprocal rank
        relevant = set(str(did) for did, rel in qrels[qid].items() if rel > 0)
        
        for rank, pid in enumerate(top_pids, 1):
            if str(doc_ids[pid]) in relevant:
                mrr_sum += 1.0 / rank
                break
        
        num_queries += 1
    
    elapsed = time.time() - start_time
    mrr_10 = mrr_sum / num_queries if num_queries > 0 else 0.0
    
    print(f"    MRR@10: {mrr_10*100:.2f}% ({num_queries} queries, {elapsed:.1f}s)")
    
    return {
        "mrr@10": mrr_10,
        "num_queries": num_queries,
        "time_seconds": elapsed
    }


def load_msmarco_dev(sample_size: int = None) -> tuple:
    """Load MS-MARCO dev queries and qrels."""
    
    queries_path = Path("data/msmarco-passage/queries.dev.small.tsv")
    qrels_path = Path("data/msmarco-passage/qrels.dev.small.tsv")
    
    # Load queries
    queries = {}
    with open(queries_path) as f:
        for line in f:
            qid, text = line.strip().split('\t')
            queries[qid] = text
    
    # Load qrels
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel
    
    # Sample if requested
    if sample_size and sample_size < len(queries):
        np.random.seed(42)
        sampled_qids = np.random.choice(list(queries.keys()), size=sample_size, replace=False)
        queries = {qid: queries[qid] for qid in sampled_qids}
    
    return queries, qrels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=500, help="Number of queries to sample (0=all)")
    parser.add_argument("--output", type=str, default="results/18_plaid_parameter_sweep.json")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PLAID Parameter Sweep for ConstBERT")
    print("=" * 60)
    
    index_path = "indices/msmarco-passage/constbert_plaid_index"
    
    # Load data
    print(f"\nLoading MS-MARCO dev (sample={args.sample})...")
    queries, qrels = load_msmarco_dev(sample_size=args.sample if args.sample > 0 else None)
    print(f"  Loaded {len(queries)} queries")
    
    # Load doc_ids from index
    doc_ids = np.load(os.path.join(index_path, "doc_ids.npy"))
    print(f"  Loaded {len(doc_ids)} document IDs")
    
    # Initialize ConstBERT for query encoding
    print("\nInitializing ConstBERT...")
    constbert = ConstBERTWrapper(model_name="pinecone/ConstBERT", batch_size=128)
    
    # Encode all queries
    query_ids = sorted(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    print(f"Encoding {len(query_ids)} queries...")
    Q_all = constbert.encode_queries(query_texts, batch_size=128)
    Q_all = torch.from_numpy(Q_all).half()
    
    if torch.cuda.is_available():
        Q_all = Q_all.cuda()
    
    print(f"  Query embeddings: {Q_all.shape}")
    
    # Initialize IndexScorer
    print(f"\nInitializing IndexScorer...")
    use_gpu = torch.cuda.is_available()
    ranker = IndexScorer(index_path, use_gpu=use_gpu, load_index_with_mmap=False)
    print(f"✅ IndexScorer initialized (GPU: {use_gpu})")
    
    # Parameter grid
    ncells_values = [4, 8, 16, 32, 64]
    threshold_values = [0.3, 0.4, 0.5]
    
    results = {
        "experiment": "PLAID Parameter Sweep",
        "sample_size": args.sample,
        "paper_reported": 0.3904,
        "sweep_results": [],
        "best_config": None,
        "best_mrr10": 0.0
    }
    
    print(f"\nRunning sweep: ncells ∈ {ncells_values}, threshold ∈ {threshold_values}")
    
    for ncells in ncells_values:
        for threshold in threshold_values:
            try:
                result = run_plaid_search(
                    ranker, Q_all, query_ids, qrels, doc_ids,
                    ncells=ncells,
                    threshold=threshold,
                    ndocs=8192,
                    k=10
                )
                
                entry = {
                    "ncells": ncells,
                    "threshold": threshold,
                    "mrr@10": result["mrr@10"],
                    "time_seconds": result["time_seconds"]
                }
                results["sweep_results"].append(entry)
                
                if result["mrr@10"] > results["best_mrr10"]:
                    results["best_mrr10"] = result["mrr@10"]
                    results["best_config"] = {"ncells": ncells, "threshold": threshold}
                    
            except Exception as e:
                print(f"    ERROR: {e}")
                results["sweep_results"].append({
                    "ncells": ncells,
                    "threshold": threshold,
                    "error": str(e)
                })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: PLAID Parameter Sweep")
    print(f"{'='*60}")
    print(f"Paper reported: {results['paper_reported']*100:.2f}%")
    print(f"Best achieved:  {results['best_mrr10']*100:.2f}% (ncells={results['best_config']['ncells']}, threshold={results['best_config']['threshold']})")
    print(f"Gap:            {(results['paper_reported'] - results['best_mrr10'])*100:.2f}%")
    
    print(f"\n{'ncells':<8} {'thresh':<8} {'MRR@10':<10}")
    print("-" * 30)
    for entry in results["sweep_results"]:
        if "error" not in entry:
            print(f"{entry['ncells']:<8} {entry['threshold']:<8} {entry['mrr@10']*100:>6.2f}%")
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
