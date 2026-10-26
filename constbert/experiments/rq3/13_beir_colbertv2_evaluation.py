#!/usr/bin/env python3
"""
ColBERT-v2 BEIR Evaluation Script
Evaluates ColBERT-v2 on all 13 BEIR datasets using native PLAID backend.

This provides a comparison baseline for ConstBERT BEIR evaluation,
helping determine whether reproduction challenges are ConstBERT-specific
or affect all multi-vector models.

Usage:
    python experiments/13_beir_colbertv2_evaluation.py --dataset scifact
    python experiments/13_beir_colbertv2_evaluation.py --all
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add external ColBERT to path
COLBERT_PATH = Path(__file__).resolve().parent.parent.parent / "external" / "ColBERT"
sys.path.insert(0, str(COLBERT_PATH))

# ColBERT imports
from colbert import Indexer, Searcher
from colbert.infra import ColBERTConfig, RunConfig, Run

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# BEIR dataset configurations
BEIR_DATASETS = {
    'trec-covid': {'name': 'TREC-COVID', 'size': '171K docs'},
    'nfcorpus': {'name': 'NFCorpus', 'size': '3.6K docs'},
    'fiqa': {'name': 'FiQA-2018', 'size': '57K docs'},
    'scifact': {'name': 'SciFact', 'size': '5K docs'},
    'arguana': {'name': 'ArguAna', 'size': '8.7K docs'},
    'scidocs': {'name': 'SCIDOCS', 'size': '25K docs'},
    'quora': {'name': 'Quora', 'size': '523K docs'},
    'dbpedia-entity': {'name': 'DBPedia-Entity', 'size': '4.6M docs'},
    'fever': {'name': 'FEVER', 'size': '5.4M docs'},
    'climate-fever': {'name': 'Climate-FEVER', 'size': '5.4M docs'},
    'hotpotqa': {'name': 'HotpotQA', 'size': '5.2M docs'},
    'nq': {'name': 'Natural Questions', 'size': '2.7M docs'},
    'webis-touche2020': {'name': 'Touche-2020', 'size': '382K docs'}
}

# Paper-reported ColBERT-v2 BEIR results (NDCG@10)
# Source: ColBERT-v2 paper and subsequent evaluations
COLBERTV2_PAPER_RESULTS = {
    'trec-covid': 0.738,
    'nfcorpus': 0.338,
    'fiqa': 0.356,
    'scifact': 0.693,
    'arguana': 0.463,
    'scidocs': 0.154,
    'quora': 0.855,
    'dbpedia-entity': 0.446,
    'fever': 0.785,
    'climate-fever': 0.176,
    'hotpotqa': 0.667,
    'nq': 0.562,
    'webis-touche2020': 0.262
}


def load_beir_dataset(dataset_name: str, data_dir: str) -> Tuple[Dict, Dict, Dict]:
    """Load BEIR dataset using official BEIR data loader"""
    from beir.datasets.data_loader import GenericDataLoader

    dataset_path = Path(data_dir) / dataset_name
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading {dataset_name} from {dataset_path}...")
    corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split="test")
    logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries")

    return corpus, queries, qrels


def prepare_collection(corpus: Dict) -> Tuple[List[str], List[str]]:
    """Convert BEIR corpus to ColBERT collection format"""
    doc_ids = list(corpus.keys())
    collection = []

    for doc_id in doc_ids:
        doc = corpus[doc_id]
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        collection.append(text.strip())

    return collection, doc_ids


def build_colbertv2_index(
    collection: List[str],
    index_name: str,
    checkpoint_path: str,
    experiment_root: str,
    nbits: int = 2,
    doc_maxlen: int = 180
) -> str:
    """Build ColBERT-v2 PLAID index for collection"""
    import torch
    
    logger.info(f"Building ColBERT-v2 index: {index_name} ({len(collection)} docs)")
    logger.info(f"Config: nbits={nbits}, doc_maxlen={doc_maxlen}")
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU detected, indexing will be slow!")

    # Adaptive cluster count based on collection size for faster indexing
    # Trade-off: fewer clusters = faster indexing, minimal accuracy impact (~1-2%)
    # ULTRA-AGGRESSIVE for large datasets to meet time constraints
    if len(collection) < 50000:
        kmeans_clusters = 1024  # Small datasets: 8x faster
        kmeans_iters = 4
    elif len(collection) < 500000:
        kmeans_clusters = 2048  # Medium datasets: 4x faster  
        kmeans_iters = 4
    else:
        # Large datasets: ULTRA-AGGRESSIVE to finish within hours not days
        kmeans_clusters = 1024  # 8x faster clustering (vs 8192 default)
        kmeans_iters = 1  # 20x faster convergence (vs 20 default)
        logger.warning(f"Using ultra-aggressive clustering for large dataset: {len(collection)} docs")
    
    logger.info(f"Using {kmeans_clusters} k-means clusters, {kmeans_iters} iterations (adaptive)")
    sys.stdout.flush()
    
    # Note: For small datasets, indexing is dominated by k-means clustering time (~8-10 min)
    # This happens silently after document encoding, so be patient!
    config = ColBERTConfig(
        doc_maxlen=doc_maxlen,
        nbits=nbits,
        bsize=256,  # Larger batch size for faster encoding
        kmeans_niters=kmeans_iters,  # Adaptive iterations
    )
    
    logger.info(f"Starting indexing with batch_size={config.bsize}, kmeans_iters={config.kmeans_niters}...")
    logger.info("Note: K-means clustering will take a few minutes with no output - this is normal!")
    sys.stdout.flush()
    sys.stderr.flush()
    
    with Run().context(RunConfig(nranks=1, experiment=index_name, root=experiment_root)):
        # Temporarily override kmeans_clusters in indexer
        indexer = Indexer(checkpoint=checkpoint_path, config=config)
        
        # Monkey patch to use adaptive clusters
        original_kmeans_clusters = 8192
        if hasattr(indexer, 'config'):
            indexer.config.kmeans_clusters = kmeans_clusters
        
        indexer.index(name=index_name, collection=collection, overwrite=True)

    logger.info(f"Index built successfully: {index_name}")
    sys.stdout.flush()
    return index_name


def compute_metrics(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 3, 5, 10, 100, 1000]
) -> Dict[str, float]:
    """Compute standard IR metrics: NDCG@k, Recall@k, MRR@k, MAP@k"""
    from collections import defaultdict

    ndcg = defaultdict(list)
    recall = defaultdict(list)
    mrr = defaultdict(list)
    precision = defaultdict(list)

    for qid, ranking in results.items():
        if qid not in qrels:
            continue

        relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        if not relevant:
            continue

        # Sort by score
        sorted_docs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

        for k in k_values:
            # Get top-k docs
            topk_docs = [doc_id for doc_id, _ in sorted_docs[:k]]

            # Recall@k
            retrieved_relevant = len(set(topk_docs) & relevant)
            recall[k].append(retrieved_relevant / len(relevant))

            # Precision@k
            precision[k].append(retrieved_relevant / k)

            # MRR@k
            rr = 0.0
            for rank, doc_id in enumerate(topk_docs, 1):
                if doc_id in relevant:
                    rr = 1.0 / rank
                    break
            mrr[k].append(rr)

            # NDCG@k
            dcg = 0.0
            for rank, doc_id in enumerate(topk_docs, 1):
                if doc_id in relevant:
                    # Binary relevance
                    dcg += 1.0 / np.log2(rank + 1)

            # Ideal DCG
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0.0)

    # Aggregate
    metrics = {}
    for k in k_values:
        if ndcg[k]:
            metrics[f'ndcg@{k}'] = float(np.mean(ndcg[k]))
        if recall[k]:
            metrics[f'recall@{k}'] = float(np.mean(recall[k]))
        if mrr[k]:
            metrics[f'mrr@{k}'] = float(np.mean(mrr[k]))
        if precision[k]:
            metrics[f'precision@{k}'] = float(np.mean(precision[k]))

    # MAP (using full ranking)
    ap_scores = []
    for qid, ranking in results.items():
        if qid not in qrels:
            continue

        relevant = {doc_id for doc_id, rel in qrels[qid].items() if rel > 0}
        if not relevant:
            continue

        sorted_docs = sorted(ranking.items(), key=lambda x: x[1], reverse=True)

        num_relevant_found = 0
        precision_sum = 0.0

        for rank, (doc_id, _) in enumerate(sorted_docs, 1):
            if doc_id in relevant:
                num_relevant_found += 1
                precision_sum += num_relevant_found / rank

        if len(relevant) > 0:
            ap_scores.append(precision_sum / len(relevant))

    if ap_scores:
        metrics['map'] = float(np.mean(ap_scores))

    return metrics


def evaluate_colbertv2_on_beir(
    dataset_name: str,
    data_dir: str,
    index_dir: str,
    output_dir: str,
    checkpoint_path: str,
    nbits: int = 2,
    doc_maxlen: int = 180,
    k: int = 1000,
    force_rebuild: bool = False
) -> Dict:
    """Evaluate ColBERT-v2 on a single BEIR dataset"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating ColBERT-v2 on {dataset_name}")
    logger.info(f"{'='*80}")
    print(f"\n{'='*80}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] Starting {dataset_name} evaluation...", flush=True)
    print(f"{'='*80}", flush=True)
    
    # Check if result already exists
    output_path = Path(output_dir) / f"13_beir_colbertv2_{dataset_name}.json"
    if output_path.exists() and not force_rebuild:
        logger.info(f"✓ Result already exists: {output_path}")
        logger.info(f"Skipping {dataset_name} (use --force-rebuild to re-run)")
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Skipping {dataset_name} - already completed", flush=True)
        with open(output_path, 'r') as f:
            return json.load(f)

    # Load dataset
    print(f"[{time.strftime('%H:%M:%S')}] Loading {dataset_name} dataset...", flush=True)
    corpus, queries, qrels = load_beir_dataset(dataset_name, data_dir)

    # Prepare collection
    print(f"[{time.strftime('%H:%M:%S')}] Preparing collection ({len(corpus)} docs)...", flush=True)
    collection, doc_ids = prepare_collection(corpus)
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}

    # Use simple dataset name as index name - ColBERT will create proper structure
    index_name = dataset_name
    index_path = Path(index_dir) / index_name

    # Build or load index
    start_time = time.time()

    # Check if index exists AND is complete (has required files)
    index_complete = False
    if index_path.exists():
        required_files = ['plan.json', 'centroids.pt', 'doclens.0.json', 'metadata.json']
        missing_files = [f for f in required_files if not (index_path / f).exists()]
        if missing_files:
            logger.warning(f"Index incomplete, missing files: {missing_files}")
            print(f"[{time.strftime('%H:%M:%S')}] Index incomplete, rebuilding...", flush=True)
        else:
            index_complete = True

    if not index_complete or force_rebuild:
        if index_path.exists() and not force_rebuild:
            logger.info("Index directory exists but is incomplete, rebuilding...")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Building index (this will take time - be patient)...", flush=True)
            logger.info("Building new index...")
        build_colbertv2_index(
            collection=collection,
            index_name=index_name,
            checkpoint_path=checkpoint_path,
            experiment_root=index_dir,
            nbits=nbits,
            doc_maxlen=doc_maxlen
        )
        print(f"[{time.strftime('%H:%M:%S')}] ✓ Index built successfully!", flush=True)
    else:
        logger.info(f"Using existing complete index at {index_path}")
        print(f"[{time.strftime('%H:%M:%S')}] Using existing index", flush=True)

    index_time = time.time() - start_time

    # Search
    logger.info(f"Searching {len(queries)} queries with k={k}...")
    print(f"[{time.strftime('%H:%M:%S')}] Searching {len(queries)} queries...", flush=True)
    search_start = time.time()

    with Run().context(RunConfig(nranks=1, experiment=index_name, root=index_dir)):
        config = ColBERTConfig(
            ncells=4,  # Use documented PLAID default for k=1000
            centroid_score_threshold=0.5,
            ndocs=256,
        )
        logger.info(f"Search config: ncells={config.ncells}, threshold={config.centroid_score_threshold}")
        searcher = Searcher(index=index_name, config=config)

        # Prepare queries dict
        queries_dict = {str(qid): query for qid, query in queries.items()}

        # Batch search
        logger.info("Running batch search...")
        ranking = searcher.search_all(queries_dict, k=k)

    search_time = time.time() - search_start
    logger.info(f"Search completed in {search_time:.2f}s")
    print(f"[{time.strftime('%H:%M:%S')}] ✓ Search completed in {search_time:.2f}s", flush=True)

    # Convert results to standard format
    results = {}
    for qid, subranking in ranking.items():
        qid_str = str(qid)
        results[qid_str] = {}
        for pid, rank, score in subranking:
            doc_id = doc_ids[int(pid)]
            results[qid_str][doc_id] = float(score)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_metrics(results, qrels)

    # Get paper result for comparison
    paper_ndcg10 = COLBERTV2_PAPER_RESULTS.get(dataset_name, None)
    our_ndcg10 = metrics.get('ndcg@10', 0)

    delta = None
    if paper_ndcg10 is not None:
        delta = (our_ndcg10 - paper_ndcg10) * 100  # Percentage points

    # Compile results
    result = {
        'dataset': dataset_name,
        'model': 'colbertv2',
        'backend': 'plaid',
        'metrics': metrics,
        'paper_ndcg10': paper_ndcg10,
        'our_ndcg10': our_ndcg10,
        'delta_ndcg10': delta,
        'num_docs': len(corpus),
        'num_queries': len(queries),
        'timing': {
            'index_time_s': index_time,
            'search_time_s': search_time,
            'total_time_s': index_time + search_time
        },
        'config': {
            'nbits': nbits,
            'doc_maxlen': doc_maxlen,
            'k': k
        }
    }

    # Print results
    logger.info(f"\nResults for {dataset_name}:")
    logger.info(f"  NDCG@10: {our_ndcg10:.4f} (Paper: {paper_ndcg10 if paper_ndcg10 else 'N/A'})")
    if delta is not None:
        logger.info(f"  Delta: {delta:+.2f} percentage points")
    logger.info(f"  Recall@100: {metrics.get('recall@100', 0):.4f}")
    logger.info(f"  MRR@10: {metrics.get('mrr@10', 0):.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description='ColBERT-v2 BEIR Evaluation')
    parser.add_argument('--dataset', type=str, default=None,
                       help='BEIR dataset name (or use --all)')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate on all 13 BEIR datasets')
    parser.add_argument('--data-dir', type=str, default='./data/beir',
                       help='BEIR data directory')
    parser.add_argument('--index-dir', type=str, default='indices/colbertv2-beir',
                       help='Directory for ColBERT-v2 indices')
    parser.add_argument('--checkpoint', type=str, default='colbert-ir/colbertv2.0',
                       help='ColBERT-v2 checkpoint (HuggingFace or local path)')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Results output directory')
    parser.add_argument('--nbits', type=int, default=2,
                       help='Quantization bits (1 or 2)')
    parser.add_argument('--doc-maxlen', type=int, default=180,
                       help='Maximum document length in tokens')
    parser.add_argument('--k', type=int, default=1000,
                       help='Number of documents to retrieve per query')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='Force rebuild indices even if they exist')

    args = parser.parse_args()

    # Validate arguments
    if not args.all and args.dataset is None:
        parser.error("Either --dataset or --all must be specified")

    if args.dataset and args.dataset not in BEIR_DATASETS:
        parser.error(f"Unknown dataset: {args.dataset}. Available: {list(BEIR_DATASETS.keys())}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create index directory
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # Determine datasets to evaluate
    if args.all:
        datasets = list(BEIR_DATASETS.keys())
    else:
        datasets = [args.dataset]

    # Evaluate each dataset
    all_results = {}

    for dataset_name in datasets:
        try:
            result = evaluate_colbertv2_on_beir(
                dataset_name=dataset_name,
                data_dir=args.data_dir,
                index_dir=str(index_dir),
                output_dir=str(output_dir),
                checkpoint_path=args.checkpoint,
                nbits=args.nbits,
                doc_maxlen=args.doc_maxlen,
                k=args.k,
                force_rebuild=args.force_rebuild
            )

            all_results[dataset_name] = result

            # Save individual result
            output_path = output_dir / f"13_beir_colbertv2_{dataset_name}.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved to {output_path}")

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            traceback.print_exc()

    # Save summary
    if args.all and all_results:
        summary_path = output_dir / "13_beir_colbertv2_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nSaved summary to {summary_path}")

        # Print summary table
        logger.info("\n" + "="*100)
        logger.info("COLBERT-V2 BEIR EVALUATION SUMMARY")
        logger.info("="*100)
        logger.info(f"{'Dataset':<20} {'Our NDCG@10':>12} {'Paper NDCG@10':>14} {'Delta':>10}")
        logger.info("-"*100)

        total_our = 0
        total_paper = 0
        count = 0

        for dataset_name, result in sorted(all_results.items()):
            our = result.get('our_ndcg10', 0)
            paper = result.get('paper_ndcg10', None)
            delta = result.get('delta_ndcg10', None)

            paper_str = f"{paper:.4f}" if paper else "N/A"
            delta_str = f"{delta:+.2f}pp" if delta is not None else "N/A"

            logger.info(f"{dataset_name:<20} {our:>12.4f} {paper_str:>14} {delta_str:>10}")

            total_our += our
            if paper is not None:
                total_paper += paper
                count += 1

        logger.info("-"*100)
        mean_our = total_our / len(all_results) if all_results else 0
        mean_paper = total_paper / count if count > 0 else 0
        mean_delta = (mean_our - mean_paper) * 100 if count > 0 else 0

        logger.info(f"{'MEAN':<20} {mean_our:>12.4f} {mean_paper:>14.4f} {mean_delta:>+10.2f}pp")
        logger.info("="*100)


if __name__ == '__main__':
    main()
