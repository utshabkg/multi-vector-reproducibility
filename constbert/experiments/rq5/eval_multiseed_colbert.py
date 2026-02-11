#!/usr/bin/env python3
"""
Multi-seed evaluation for fine-tuned ColBERT models on ToT TEST set.
Converts checkpoints, builds indices, evaluates, and aggregates results.

Usage:
    python experiments/eval_multiseed_colbert.py --seed 42
    python experiments/eval_multiseed_colbert.py --seed 123
    python experiments/eval_multiseed_colbert.py --seed 456
    python experiments/eval_multiseed_colbert.py --aggregate
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
import numpy as np
import torch
import subprocess

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COLBERT_COPY = PROJECT_ROOT / "../colbertv2"
BASE_CHECKPOINT = Path("colbert-ir/colbertv2.0")
INDICES_ROOT = Path("indices/trec-tot-2025")
CORPUS_JSONL = Path("data/trec-tot-2025/trec-tot-2025-corpus.jsonl")
TEST_QUERIES = Path("data/trec-tot-2025/queries/test-2025-queries.jsonl")
TEST_QRELS = Path("data/trec-tot-2025/qrel/test-2025-qrel.txt")


def convert_checkpoint_to_hf(seed: int) -> Path:
    """Convert .pt checkpoint to HuggingFace format."""
    
    checkpoint_dir = COLBERT_COPY / f"colbert-replicability/colbert/models/colbertv2-tot-seed{seed}"
    pt_file = checkpoint_dir / "best_model.pt"
    hf_dir = INDICES_ROOT / f"colbertv2-fine-tot-seed{seed}-hf"
    
    if hf_dir.exists() and (hf_dir / "pytorch_model.bin").exists():
        logger.info(f"HF checkpoint already exists at {hf_dir}")
        return hf_dir
    
    logger.info(f"Converting {pt_file} to HF format...")
    
    # Create HF directory by copying base checkpoint structure
    hf_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy config and tokenizer files from base checkpoint
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json", 
                  "special_tokens_map.json", "vocab.txt", "artifact.metadata"]:
        src = BASE_CHECKPOINT / fname
        dst = hf_dir / fname
        if src.exists():
            shutil.copy(src, dst)
    
    # Load the .pt checkpoint and extract model weights
    checkpoint = torch.load(pt_file, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Save as pytorch_model.bin
    torch.save(state_dict, hf_dir / "pytorch_model.bin")
    
    logger.info(f"Saved HF checkpoint to {hf_dir}")
    return hf_dir


def build_plaid_index(seed: int, hf_checkpoint: Path) -> Path:
    """Build PLAID index using the fine-tuned checkpoint."""
    
    index_name = f"trec-tot-2025-colbertv2-fine-seed{seed}"
    index_path = INDICES_ROOT / index_name
    
    if index_path.exists() and (index_path / "0.residuals.pt").exists():
        logger.info(f"Index already exists at {index_path}")
        return index_path
    
    logger.info(f"Building PLAID index for seed {seed}...")
    logger.info(f"  Checkpoint: {hf_checkpoint}")
    logger.info(f"  Index: {index_path}")
    
    # Use the indexing script
    script_path = COLBERT_COPY / "colbert-replicability/colbert/scripts/index_colbertv2_tot.sh"
    
    cmd = [
        "bash", str(script_path),
        str(CORPUS_JSONL),
        str(hf_checkpoint),
        index_name,
        str(INDICES_ROOT)
    ]
    
    logger.info(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Indexing failed: {result.stderr}")
        raise RuntimeError(f"Indexing failed for seed {seed}")
    
    elapsed = time.time() - start_time
    logger.info(f"Indexing completed in {elapsed/60:.1f} minutes")
    
    return index_path


def evaluate_index(seed: int, index_path: Path) -> dict:
    """Evaluate the index on ToT TEST set."""
    
    results_dir = PROJECT_ROOT / "results" / "multiseed"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"colbert_seed{seed}_eval.json"
    
    if results_path.exists():
        logger.info(f"Results already exist at {results_path}")
        with open(results_path) as f:
            return json.load(f)
    
    logger.info(f"Evaluating seed {seed} on TEST set...")
    
    # Import ColBERT searcher
    # ColBERT should be installed via: pip install colbert-ai
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    
    # Load queries
    queries = {}
    with open(TEST_QUERIES) as f:
        for line in f:
            doc = json.loads(line)
            queries[str(doc['query_id'])] = doc['query']
    
    # Load qrels
    qrels = {}
    with open(TEST_QRELS) as f:
        for line in f:
            parts = line.strip().split()
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    
    logger.info(f"  Loaded {len(queries)} queries, {len(qrels)} qrels")
    
    # Initialize searcher
    hf_checkpoint = INDICES_ROOT / f"colbertv2-fine-tot-seed{seed}-hf"
    
    with Run().context(RunConfig(nranks=1, experiment=f"eval_seed{seed}")):
        config = ColBERTConfig(root=str(INDICES_ROOT))
        searcher = Searcher(
            index=str(index_path.name),
            index_root=str(INDICES_ROOT),
            checkpoint=str(hf_checkpoint),
            config=config
        )
        
        # Load ID mapping
        idmap_path = COLBERT_COPY / f"colbert-replicability/colbert/tmp/{index_path.name}_idmap.tsv"
        internal_to_original = {}
        if idmap_path.exists():
            with open(idmap_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        internal_to_original[int(parts[0])] = parts[1]
        
        # Search and evaluate
        all_results = {}
        
        for qid, query in queries.items():
            results = searcher.search(query, k=1000)
            
            ranked_docs = {}
            for rank, (pid, score) in enumerate(zip(results[0], results[2])):
                # Map internal ID to original doc ID
                original_pid = internal_to_original.get(pid, str(pid))
                ranked_docs[original_pid] = float(score)
            
            all_results[qid] = ranked_docs
    
    # Compute metrics
    def compute_mrr(results, qrels, k=10):
        mrr_sum = 0.0
        count = 0
        for qid in qrels:
            if qid in results:
                for rank, (doc_id, _) in enumerate(sorted(results[qid].items(), key=lambda x: -x[1])[:k]):
                    if doc_id in qrels[qid] and qrels[qid][doc_id] > 0:
                        mrr_sum += 1.0 / (rank + 1)
                        break
                count += 1
        return mrr_sum / count if count > 0 else 0.0
    
    def compute_recall(results, qrels, k=1000):
        recall_sum = 0.0
        count = 0
        for qid in qrels:
            if qid in results:
                retrieved = set(list(results[qid].keys())[:k])
                relevant = set(doc_id for doc_id, rel in qrels[qid].items() if rel > 0)
                if len(relevant) > 0:
                    recall_sum += len(retrieved & relevant) / len(relevant)
                    count += 1
        return recall_sum / count if count > 0 else 0.0
    
    mrr_10 = compute_mrr(all_results, qrels, k=10)
    recall_1000 = compute_recall(all_results, qrels, k=1000)
    
    logger.info(f"\n=== RESULTS for ColBERT seed={seed} ===")
    logger.info(f"  MRR@10: {mrr_10*100:.2f}%")
    logger.info(f"  Recall@1000: {recall_1000*100:.2f}%")
    
    results = {
        "model": "colbert",
        "seed": seed,
        "metrics": {
            "MRR@10": float(mrr_10),
            "Recall@1000": float(recall_1000)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    return results


def evaluate_seed(seed: int):
    """Full evaluation pipeline for one seed."""
    logger.info("=" * 80)
    logger.info(f"Evaluating ColBERT seed={seed}")
    logger.info("=" * 80)
    
    # Step 1: Convert checkpoint
    hf_checkpoint = convert_checkpoint_to_hf(seed)
    
    # Step 2: Build index
    index_path = build_plaid_index(seed, hf_checkpoint)
    
    # Step 3: Evaluate
    results = evaluate_index(seed, index_path)
    
    return results


def aggregate_results():
    """Aggregate results from all seeds."""
    results_dir = PROJECT_ROOT / "results" / "multiseed"
    seeds = [42, 123, 456]
    
    metrics_list = {"MRR@10": [], "Recall@1000": []}
    
    for seed in seeds:
        results_path = results_dir / f"colbert_seed{seed}_eval.json"
        if results_path.exists():
            with open(results_path) as f:
                data = json.load(f)
                for metric in metrics_list:
                    if metric in data["metrics"]:
                        metrics_list[metric].append(data["metrics"][metric])
    
    if metrics_list["MRR@10"]:
        logger.info(f"\nCOLBERT (n={len(metrics_list['MRR@10'])}):")
        for metric, values in metrics_list.items():
            if values:
                mean = np.mean(values) * 100
                std = np.std(values) * 100
                logger.info(f"  {metric}: {mean:.2f}% ± {std:.2f}%")
    else:
        logger.info("No ColBERT results found yet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 123, 456])
    parser.add_argument("--aggregate", action="store_true")
    
    args = parser.parse_args()
    
    if args.aggregate:
        aggregate_results()
    elif args.seed:
        evaluate_seed(args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
