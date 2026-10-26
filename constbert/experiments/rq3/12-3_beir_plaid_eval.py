#!/usr/bin/env python3
"""
Evaluate BEIR datasets using ColBERT's IndexScorer with PLAID indices and ConstBERT embeddings.

This script attempts exact reproduction of the paper's BEIR evaluation by:
1. Using ColBERT's IndexScorer (same infrastructure as authors)
2. Using PLAID index format with IVF (same as authors)
3. Using ConstBERT for query encoding
4. Using ranker.rank() for PLAID search

This will demonstrate whether PLAID incompatibility extends to out-of-domain tasks.

Usage:
    # Evaluate single dataset
    python experiments/12-3_beir_plaid_eval.py --dataset scifact

    # Evaluate all 5 priority datasets
    python experiments/12-3_beir_plaid_eval.py --all
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
from beir.datasets.data_loader import GenericDataLoader
from models.constbert_wrapper import ConstBERTWrapper
from evaluation.metrics import evaluate_retrieval

# Priority BEIR datasets
BEIR_DATASETS = ['arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa', 'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact', 'trec-covid', 'webis-touche2020']


def load_beir_data(dataset_name: str, data_dir: str = "./data/beir"):
    """Load BEIR dataset queries and qrels"""
    dataset_path = os.path.join(data_dir, dataset_name)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    print(f"Loading {dataset_name} dataset...")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    print(f"  - {len(corpus):,} documents")
    print(f"  - {len(queries):,} queries")
    print(f"  - {len(qrels):,} query-doc pairs")

    return queries, qrels


def evaluate_beir_with_plaid(dataset_name: str,
                             index_root: str,
                             data_dir: str,
                             k: int = 1000,
                             batch_size: int = 128,
                             output_dir: str = "results"):
    """
    Evaluate BEIR dataset using PLAID index

    Args:
        dataset_name: Name of BEIR dataset (e.g., 'scifact')
        index_root: Root directory containing PLAID indices
        data_dir: Directory containing BEIR data
        k: Number of results per query
        batch_size: Query encoding batch size
        output_dir: Directory to save results

    Returns:
        Dictionary of evaluation metrics
    """
    print("="*80)
    print(f"BEIR Evaluation with PLAID: {dataset_name.upper()}")
    print("="*80)

    # Load BEIR data
    queries_dict, qrels = load_beir_data(dataset_name, data_dir)

    # Load PLAID index
    index_name = f'{dataset_name}_constbert_plaid'
    index_path = os.path.join(index_root, index_name)

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"PLAID index not found at {index_path}")

    print(f"\nIndex: {index_path}")

    # Load doc_ids from index
    doc_ids = np.load(os.path.join(index_path, "doc_ids.npy"))
    print(f"  - {len(doc_ids):,} documents in index")

    # Initialize ConstBERT for query encoding
    print("\nInitializing ConstBERT...")
    constbert = ConstBERTWrapper(model_name="pinecone/ConstBERT", batch_size=batch_size)

    # Encode all queries with ConstBERT
    print(f"\nEncoding {len(queries_dict)} queries with ConstBERT...")
    query_ids = sorted(queries_dict.keys())
    query_texts = [queries_dict[qid] for qid in query_ids]

    Q_all = constbert.encode_queries(query_texts, batch_size=batch_size)
    Q_all = torch.from_numpy(Q_all).half()  # (num_queries, num_tokens, 128), float16
    print(f"  Query embeddings: {Q_all.shape}")

    # Initialize ColBERT's IndexScorer with PLAID index
    print(f"\nInitializing IndexScorer...")

    use_gpu = torch.cuda.is_available()
    ranker = IndexScorer(index_path, use_gpu=use_gpu, load_index_with_mmap=False)

    print(f"✅ IndexScorer initialized")
    print(f"   GPU: {use_gpu}")

    # Configure search parameters
    config = ColBERTConfig()
    if k <= 10:
        config.ncells = 1
        config.centroid_score_threshold = 0.5
        config.ndocs = 256
    elif k <= 100:
        config.ncells = 2
        config.centroid_score_threshold = 0.45
        config.ndocs = 1024
    else:
        config.ncells = 4
        config.centroid_score_threshold = 0.4
        config.ndocs = max(k * 4, 4096)

    print(f"   Search config: ncells={config.ncells}, ndocs={config.ndocs}, threshold={config.centroid_score_threshold}")

    # Search with pre-computed ConstBERT query embeddings
    print(f"\nSearching with PLAID (k={k})...")
    results = {}
    start_time = time.time()

    if use_gpu:
        Q_all = Q_all.cuda()

    for idx, qid in enumerate(tqdm(query_ids, desc="Searching")):
        # Get query embedding for this query
        Q = Q_all[idx:idx+1]  # (1, num_tokens, 128)

        # Use IndexScorer.rank() for PLAID search
        pids, scores = ranker.rank(config, Q)

        # Take top-k
        top_pids = pids[:k].cpu().numpy() if hasattr(pids, 'cpu') else pids[:k]
        top_scores = scores[:k].cpu().tolist() if hasattr(scores, 'cpu') else list(scores[:k])

        # Convert passage indices to document IDs with scores
        results[qid] = {str(doc_ids[pid]): score for pid, score in zip(top_pids, top_scores)}

    search_time = time.time() - start_time
    avg_time_ms = search_time / len(query_ids) * 1000

    print(f"✅ Search completed in {search_time:.2f}s ({avg_time_ms:.1f}ms per query)")

    # Evaluate using BEIR-style metrics
    print("\nComputing metrics...")
    metrics = evaluate_retrieval(results, qrels)

    # Extract key metrics (metrics use @ format, not _at_)
    key_metrics = {
        'ndcg@10': metrics.get('ndcg@10', 0.0),
        'recall@100': metrics.get('recall@100', 0.0),
        'recall@1000': metrics.get('recall@1000', 0.0),
        'map@100': metrics.get('map@100', metrics.get('map@1000', 0.0)),  # Fallback to map@1000 if map@100 not available
    }

    # Print results
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"  Dataset: {dataset_name}")
    print(f"  ncells={config.ncells}, ndocs={config.ndocs}, threshold={config.centroid_score_threshold}")
    print(f"\n  NDCG@10:     {key_metrics['ndcg@10']:.4f} ({key_metrics['ndcg@10']*100:.2f}%)")
    print(f"  Recall@100:  {key_metrics['recall@100']:.4f}")
    print(f"  Recall@1000: {key_metrics['recall@1000']:.4f}")
    print(f"  MAP@100:     {key_metrics['map@100']:.4f}")
    print(f"\n  Avg latency: {avg_time_ms:.1f}ms per query")
    print("="*80)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'12_beir_plaid_{dataset_name}.json')

    output_data = {
        'dataset': dataset_name,
        'backend': 'PLAID',
        'metrics': key_metrics,
        'all_metrics': metrics,
        'config': {
            'index': index_name,
            'k': k,
            'ncells': config.ncells,
            'ndocs': config.ndocs,
            'centroid_score_threshold': config.centroid_score_threshold,
        },
        'timing': {
            'total_time_s': search_time,
            'avg_time_ms': avg_time_ms,
        },
        'num_queries': len(query_ids),
        'num_docs': len(doc_ids),
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return key_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate BEIR with PLAID using ColBERT infrastructure')
    parser.add_argument('--dataset', type=str, choices=BEIR_DATASETS,
                       help='BEIR dataset name')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all 5 priority datasets')
    parser.add_argument('--index_root', type=str,
                       default='indices/beir_plaid',
                       help='Root directory containing PLAID indices')
    parser.add_argument('--data_dir', type=str, default='./data/beir',
                       help='Directory containing BEIR data')
    parser.add_argument('--k', type=int, default=1000,
                       help='Number of results per query')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Query encoding batch size')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")

    # Determine which datasets to process
    if args.all:
        datasets = BEIR_DATASETS
    else:
        datasets = [args.dataset]

    print(f"\n{'='*80}")
    print(f"BEIR PLAID Evaluation")
    print(f"{'='*80}\n")
    print(f"Configuration:")
    print(f"  - Datasets: {', '.join(datasets)}")
    print(f"  - Index root: {args.index_root}")
    print(f"  - Data directory: {args.data_dir}")
    print(f"  - k: {args.k}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Output: {args.output}\n")

    # Evaluate each dataset
    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'#'*80}")
        print(f"# {dataset_name.upper()}")
        print(f"{'#'*80}\n")

        start_time = time.time()

        try:
            metrics = evaluate_beir_with_plaid(
                dataset_name=dataset_name,
                index_root=args.index_root,
                data_dir=args.data_dir,
                k=args.k,
                batch_size=args.batch_size,
                output_dir=args.output
            )

            all_results[dataset_name] = metrics

            elapsed = time.time() - start_time
            print(f"\n✅ Completed {dataset_name} in {elapsed:.1f}s\n")

        except Exception as e:
            print(f"❌ Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY: BEIR PLAID Evaluation")
        print(f"{'='*80}\n")
        print(f"{'Dataset':<15} {'NDCG@10':>10}")
        print("-" * 80)

        ndcg_values = []
        for dataset_name in datasets:
            if dataset_name in all_results:
                ndcg = all_results[dataset_name]['ndcg@10']
                ndcg_values.append(ndcg)
                print(f"{dataset_name:<15} {ndcg*100:>9.2f}%")

        if ndcg_values:
            mean_ndcg = np.mean(ndcg_values)
            print("-" * 80)
            print(f"{'Mean':<15} {mean_ndcg*100:>9.2f}%")

        print("="*80)

        # Save summary
        summary_file = os.path.join(args.output, '12_beir_plaid_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved to: {summary_file}")

    print(f"\n✅ All evaluations completed!\n")


if __name__ == "__main__":
    main()
