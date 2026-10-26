#!/usr/bin/env python3
"""
Query Length Ablation Experiment

Tests whether MaxSim performance degrades monotonically with query length.
Truncates ToT queries to different lengths and measures performance.

This isolates query length as a factor in the ToT performance collapse.

Uses full FAISS index for end-to-end retrieval (not just reranking).
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
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import compute_mrr, compute_recall, compute_ndcg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def truncate_query(query: str, max_words: int) -> str:
    """Truncate query to max_words."""
    words = query.split()
    if len(words) <= max_words:
        return query
    return ' '.join(words[:max_words])


def run_length_ablation(
    word_lengths: List[int] = [10, 20, 40, 60, 80, 100, 121],
    seed: int = 42
):
    """
    Run query length ablation experiment.
    
    For each length:
    1. Truncate ALL test queries to that length
    2. Run full end-to-end FAISS retrieval
    3. Compute MRR@10 and Recall@1000
    
    This tests the hypothesis that longer queries cause MaxSim degradation.
    """
    
    logger.info("=" * 60)
    logger.info("Query Length Ablation Experiment")
    logger.info("=" * 60)
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Paths
    index_dir = Path("indices/trec-tot-2025/constbert_tot_faiss_index")
    index_path = index_dir / "index"
    
    # Load ToT data
    logger.info("\n[1/3] Loading ToT data...")
    loader = TRECToTDataLoader()
    test_queries = loader.load_queries("test")
    test_qrels = loader.load_qrels("test")
    logger.info(f"  Loaded {len(test_queries)} test queries")
    logger.info(f"  {len(test_qrels)} queries have relevance judgments")
    
    # Analyze query lengths
    query_lengths = [len(q.split()) for q in test_queries.values()]
    logger.info(f"\n  Query length stats:")
    logger.info(f"    Mean: {np.mean(query_lengths):.1f} words")
    logger.info(f"    Median: {np.median(query_lengths):.1f} words")
    logger.info(f"    Min: {np.min(query_lengths)} words")
    logger.info(f"    Max: {np.max(query_lengths)} words")
    logger.info(f"    Std: {np.std(query_lengths):.1f} words")
    
    # Load FAISS index
    logger.info(f"\n[2/3] Loading FAISS index from {index_path}...")
    start_time = time.time()
    index = ConstBERTFAISSIndex.load(index_path)
    logger.info(f"  Index loaded in {time.time() - start_time:.1f}s")
    logger.info(f"  Index contains {index.num_docs:,} documents")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"  Loading ConstBERT model (device: {device})...")
    model = ConstBERTWrapper(device=device)
    
    # Results storage
    results = {
        "experiment": "Query Length Ablation",
        "seed": seed,
        "dataset": {
            "num_queries": len(test_queries),
            "num_with_relevance": len(test_qrels),
            "original_query_stats": {
                "mean_length": float(np.mean(query_lengths)),
                "median_length": float(np.median(query_lengths)),
                "min_length": int(np.min(query_lengths)),
                "max_length": int(np.max(query_lengths)),
                "std_length": float(np.std(query_lengths))
            }
        },
        "lengths": {}
    }
    
    # Run ablation for each length
    logger.info("\n[3/3] Running retrieval for each query length...")
    
    for max_words in word_lengths:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing max length: {max_words} words")
        logger.info(f"{'='*50}")
        
        # Truncate queries
        query_ids = list(test_queries.keys())
        truncated_queries = {
            qid: truncate_query(test_queries[qid], max_words)
            for qid in query_ids
        }
        
        # Track actual truncated lengths
        truncated_lengths = [len(q.split()) for q in truncated_queries.values()]
        actual_mean = np.mean(truncated_lengths)
        pct_truncated = sum(1 for orig, trunc in zip(query_lengths, truncated_lengths) if trunc < orig) / len(query_lengths) * 100
        
        logger.info(f"  Actual mean length after truncation: {actual_mean:.1f} words")
        logger.info(f"  Queries truncated: {pct_truncated:.1f}%")
        
        # Encode truncated queries
        logger.info("  Encoding queries...")
        query_texts = [truncated_queries[qid] for qid in query_ids]
        query_embeddings = model.encode_queries(query_texts, batch_size=128, show_progress=True)
        
        # Retrieve
        logger.info("  Retrieving...")
        all_results = {}
        
        for i, qid in enumerate(tqdm(query_ids, desc=f"Retrieving (max {max_words})")):
            doc_ids_batch, scores_batch = index.search(
                query_embeddings[i:i+1],
                k=1000,
                candidate_mult=10
            )
            
            # Get first result (single query)
            doc_ids = doc_ids_batch[0]
            scores = scores_batch[0]
            
            all_results[qid] = {
                doc_id: float(score) 
                for doc_id, score in zip(doc_ids, scores)
            }
        
        # Compute metrics
        mrr_10 = compute_mrr(all_results, test_qrels, k=10)
        recall_1000 = compute_recall(all_results, test_qrels, k=1000)
        ndcg_10 = compute_ndcg(all_results, test_qrels, k=10)
        
        logger.info(f"\n  Results for max {max_words} words:")
        logger.info(f"    MRR@10: {mrr_10*100:.2f}%")
        logger.info(f"    R@1000: {recall_1000*100:.2f}%")
        logger.info(f"    NDCG@10: {ndcg_10*100:.2f}%")
        
        results["lengths"][max_words] = {
            "mrr@10": float(mrr_10),
            "recall@1000": float(recall_1000),
            "ndcg@10": float(ndcg_10),
            "actual_mean_length": float(actual_mean),
            "pct_queries_truncated": float(pct_truncated)
        }
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query Length Ablation Experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lengths", type=str, default="10,20,40,60,80,100,121",
                        help="Comma-separated list of max word lengths to test")
    parser.add_argument("--output", type=str, default="results/15_query_length_ablation.json")
    args = parser.parse_args()
    
    word_lengths = [int(x) for x in args.lengths.split(",")]
    
    # Run experiment
    results = run_length_ablation(
        word_lengths=word_lengths,
        seed=args.seed
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY: Query Length vs Performance")
    logger.info(f"{'='*60}")
    logger.info(f"{'Max Length':<12} {'MRR@10':<10} {'R@1000':<10} {'Actual Mean':<15}")
    logger.info("-" * 50)
    for length in sorted(results["lengths"].keys(), key=lambda x: int(x)):
        data = results["lengths"][length]
        logger.info(f"{length:<12} {data['mrr@10']*100:>6.2f}%    {data['recall@1000']*100:>6.2f}%    {data['actual_mean_length']:>6.1f} words")
    
    logger.info(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
