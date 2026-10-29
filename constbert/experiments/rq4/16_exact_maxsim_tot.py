#!/usr/bin/env python3
"""
Exact MaxSim Upper Bound on ToT

Computes exact MaxSim scores (no approximation) to establish architectural upper bound.
Scores every document in the corpus for a random sample of TEST queries.

If Exact ≈ FAISS/PLAID, the problem is architectural (MaxSim).
If Exact >> FAISS/PLAID, the problem is approximation quality.
"""

import json
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.tot_loader import TRECToTDataLoader
from models.constbert_wrapper import ConstBERTWrapper
from evaluation.metrics import compute_mrr, compute_recall, compute_ndcg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_exact_maxsim(Q: torch.Tensor, D: torch.Tensor) -> float:
    """
    Compute exact MaxSim score between query Q and document D.
    
    Q: (num_query_tokens, 128)
    D: (num_doc_tokens, 128)
    
    Returns: MaxSim score (sum of max similarities)
    """
    # Normalize
    Q = Q / (Q.norm(dim=1, keepdim=True) + 1e-8)
    D = D / (D.norm(dim=1, keepdim=True) + 1e-8)
    
    # Compute similarity matrix: (num_query_tokens, num_doc_tokens)
    sim_matrix = torch.mm(Q, D.t())
    
    # MaxSim: sum of max similarity for each query token
    max_sims = sim_matrix.max(dim=1)[0]
    maxsim_score = max_sims.sum().item()
    
    return maxsim_score


def run_exact_maxsim_evaluation(num_queries: int = 100, seed: int = 42):
    """
    Run exact MaxSim evaluation on ToT.
    
    For computational feasibility, sample N queries. Each one is scored against
    every document with exact MaxSim (no approximation).
    """
    
    logger.info("=" * 60)
    logger.info("Exact MaxSim Upper Bound Evaluation")
    logger.info("=" * 60)
    
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Load data
    logger.info("\n[1/4] Loading ToT data...")
    loader = TRECToTDataLoader()
    test_queries = loader.load_queries("test")
    test_qrels = loader.load_qrels("test")
    
    # Stored ConstBERT document embeddings and doc IDs, as saved by 04_tot_faiss_eval.py
    index_dir = Path("indices/trec-tot-2025/constbert_tot_faiss_index")
    embeddings = np.load(index_dir / "embeddings.npy")  # (num_docs, 32, 128) float16
    doc_ids = np.load(index_dir / "metadata.npy", allow_pickle=True).tolist()
    
    logger.info(f"  Loaded {len(test_queries)} test queries")
    logger.info(f"  Loaded {len(doc_ids):,} documents")
    logger.info(f"  {len(test_qrels)} queries have relevance judgments")
    
    # Sample queries
    query_ids = [qid for qid in test_qrels.keys()]
    if len(query_ids) > num_queries:
        query_ids = np.random.choice(query_ids, size=num_queries, replace=False).tolist()
    
    logger.info(f"\n  Sampled {len(query_ids)} queries for evaluation")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"\n[2/4] Loading ConstBERT model (device: {device})...")
    model = ConstBERTWrapper(device=device)
    
    # Encode queries
    logger.info(f"\n[3/4] Encoding {len(query_ids)} queries...")
    query_texts = [test_queries[qid] for qid in query_ids]
    Q_all = model.encode_queries(query_texts, batch_size=128, show_progress=True)
    # Scoring runs on CPU: the document embeddings (~52 GB) do not fit in GPU memory
    Q_all = torch.from_numpy(Q_all)  # (num_queries, 32, 128)
    
    logger.info(f"  Query embeddings: {Q_all.shape}")
    
    # Use the stored embeddings of ALL documents (for exact MaxSim). They stay float16
    # in memory, and each document is cast to float32 when it is scored (exact cast)
    logger.info(f"\n  Using stored embeddings of {len(doc_ids):,} documents...")
    D_all = torch.from_numpy(embeddings)  # (num_docs, 32, 128)
    
    logger.info(f"  Document embeddings: {D_all.shape}")
    
    # Run exact MaxSim retrieval
    logger.info(f"\n[4/4] Computing exact MaxSim scores...")
    results = {}
    query_times = []
    
    for i, qid in enumerate(tqdm(query_ids, desc="Evaluating")):
        start_time = time.time()
        
        Q = Q_all[i]  # (32, 128)
        
        # Score ALL documents (exact MaxSim)
        scores = []
        for doc_idx in range(len(doc_ids)):
            D = D_all[doc_idx].float()  # (32, 128)
            score = compute_exact_maxsim(Q, D)
            scores.append(score)
        
        # Get top-1000
        top_indices = np.argsort(scores)[::-1][:1000]
        top_doc_ids = [doc_ids[idx] for idx in top_indices]
        top_scores = [scores[idx] for idx in top_indices]
        
        results[qid] = [(did, score) for did, score in zip(top_doc_ids, top_scores)]
        
        query_times.append(time.time() - start_time)
    
    mean_time = np.mean(query_times)
    logger.info(f"\n  Mean query time: {mean_time:.2f}s")
    
    # Evaluate metrics
    logger.info("\n[5/5] Computing metrics...")
    metrics = {
        "mrr@10": compute_mrr(results, test_qrels, k=10),
        "recall@1000": compute_recall(results, test_qrels, k=1000),
        "ndcg@10": compute_ndcg(results, test_qrels, k=10),
        "mean_time_seconds": mean_time
    }
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS: Exact MaxSim Upper Bound")
    logger.info("=" * 60)
    logger.info(f"  MRR@10:        {metrics['mrr@10']*100:.2f}%")
    logger.info(f"  Recall@1000:   {metrics['recall@1000']*100:.2f}%")
    logger.info(f"  NDCG@10:       {metrics['ndcg@10']*100:.2f}%")
    logger.info(f"  Mean time:     {mean_time:.2f}s per query")
    
    # Comparison with FAISS/PLAID
    logger.info("\n" + "-" * 60)
    logger.info("COMPARISON (from previous experiments):")
    logger.info("-" * 60)
    logger.info(f"  FAISS MRR@10:  4.27%")
    logger.info(f"  PLAID MRR@10:  5.66%")
    logger.info(f"  Exact MRR@10:  {metrics['mrr@10']*100:.2f}%")
    logger.info("")
    logger.info(f"  → Exact ≈ FAISS/PLAID: Problem is ARCHITECTURAL (MaxSim)")
    logger.info(f"  → Exact >> FAISS/PLAID: Problem is APPROXIMATION")
    logger.info("=" * 60)
    
    # Save results
    output = {
        "experiment": "Exact MaxSim Upper Bound",
        "seed": seed,
        "num_queries": len(query_ids),
        "num_documents": len(doc_ids),
        "metrics": metrics,
        "comparison": {
            "faiss_mrr@10": 0.0427,
            "faiss_recall@1000": 0.2588,
            "plaid_mrr@10": 0.0566,
            "plaid_recall@1000": 0.2894
        },
        "interpretation": "This establishes the architectural upper bound for MaxSim on ToT. If Exact >> FAISS/PLAID, the problem is approximation quality. If Exact ≈ FAISS/PLAID, the problem is MaxSim's uniform token weighting."
    }
    
    output_path = Path("results/16_exact_maxsim_tot.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_queries", type=int, default=100,
                       help="Number of queries to evaluate (sampled randomly)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    args = parser.parse_args()
    
    run_exact_maxsim_evaluation(
        num_queries=args.num_queries,
        seed=args.seed
    )
