"""
Evaluation metrics for information retrieval.
Implements MRR@k, Recall@k, NDCG@k, and other standard IR metrics.
"""
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict


def compute_mrr(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """
    Compute Mean Reciprocal Rank at cutoff k.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        k: Cutoff for MRR
    
    Returns:
        MRR@k score
    """
    reciprocal_ranks = []
    
    for qid, results in run.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
        
        for rank, (doc_id, _) in enumerate(results[:k], 1):
            if doc_id in relevant_docs:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def compute_recall(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 1000
) -> float:
    """
    Compute Recall at cutoff k.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        k: Cutoff for recall
    
    Returns:
        Recall@k score
    """
    recall_scores = []
    
    for qid, results in run.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue
        
        retrieved_relevant = set(doc_id for doc_id, _ in results[:k]) & relevant_docs
        recall = len(retrieved_relevant) / len(relevant_docs)
        recall_scores.append(recall)
    
    return np.mean(recall_scores) if recall_scores else 0.0


def compute_ndcg(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at cutoff k.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        k: Cutoff for NDCG
    
    Returns:
        NDCG@k score
    """
    ndcg_scores = []
    
    for qid, results in run.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        # DCG
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(results[:k], 1):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2 ** rel - 1) / np.log2(rank + 1)
        
        # IDCG
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
        idcg = sum((2 ** rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, 1))
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0.0


def compute_precision(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10
) -> float:
    """
    Compute Precision at cutoff k.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        k: Cutoff for precision
    
    Returns:
        Precision@k score
    """
    precision_scores = []
    
    for qid, results in run.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
        retrieved_relevant = sum(1 for doc_id, _ in results[:k] if doc_id in relevant_docs)
        precision = retrieved_relevant / min(k, len(results))
        precision_scores.append(precision)
    
    return np.mean(precision_scores) if precision_scores else 0.0


def compute_map(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 1000
) -> float:
    """
    Compute Mean Average Precision at cutoff k.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        k: Cutoff for MAP
    
    Returns:
        MAP@k score
    """
    ap_scores = []
    
    for qid, results in run.items():
        if qid not in qrels or not qrels[qid]:
            continue
        
        relevant_docs = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
        if not relevant_docs:
            continue
        
        num_relevant = 0
        sum_precisions = 0.0
        
        for rank, (doc_id, _) in enumerate(results[:k], 1):
            if doc_id in relevant_docs:
                num_relevant += 1
                precision_at_rank = num_relevant / rank
                sum_precisions += precision_at_rank
        
        if num_relevant > 0:
            ap_scores.append(sum_precisions / len(relevant_docs))
    
    return np.mean(ap_scores) if ap_scores else 0.0


def evaluate_retrieval(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str] = None
) -> Dict[str, float]:
    """
    Compute multiple evaluation metrics.
    
    Args:
        run: Dict mapping query_id -> [(doc_id, score), ...]
        qrels: Dict mapping query_id -> {doc_id -> relevance}
        metrics: List of metrics to compute. Default: all metrics.
    
    Returns:
        Dict mapping metric_name -> score
    """
    if metrics is None:
        metrics = [
            'mrr@10', 'recall@50', 'recall@200', 'recall@1000',
            'ndcg@10', 'ndcg@100', 'map@1000', 'p@10'
        ]
    
    results = {}
    
    for metric in metrics:
        if metric.startswith('mrr@'):
            k = int(metric.split('@')[1])
            results[metric] = compute_mrr(run, qrels, k)
        elif metric.startswith('recall@'):
            k = int(metric.split('@')[1])
            results[metric] = compute_recall(run, qrels, k)
        elif metric.startswith('ndcg@'):
            k = int(metric.split('@')[1])
            results[metric] = compute_ndcg(run, qrels, k)
        elif metric.startswith('map@'):
            k = int(metric.split('@')[1])
            results[metric] = compute_map(run, qrels, k)
        elif metric.startswith('p@'):
            k = int(metric.split('@')[1])
            results[metric] = compute_precision(run, qrels, k)
        else:
            print(f"Warning: Unknown metric {metric}")
    
    return results


def print_evaluation_results(results: Dict[str, float], title: str = "Evaluation Results"):
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    for metric, score in sorted(results.items()):
        print(f"{metric:20s}: {score:.4f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test evaluation metrics
    print("Testing evaluation metrics...")
    
    # Mock data
    qrels = {
        'q1': {'d1': 1, 'd2': 0, 'd3': 1},
        'q2': {'d4': 1, 'd5': 1},
    }
    
    run = {
        'q1': [('d1', 0.9), ('d2', 0.8), ('d3', 0.7)],
        'q2': [('d4', 0.95), ('d6', 0.85), ('d5', 0.75)],
    }
    
    # Compute metrics
    results = evaluate_retrieval(run, qrels)
    print_evaluation_results(results)
    
    print("Metric test complete!")
