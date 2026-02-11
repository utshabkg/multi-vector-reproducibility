#!/usr/bin/env python3
"""
Multi-seed evaluation for fine-tuned ConstBERT and ColBERT models on ToT TEST set.
Evaluates 3 seeds per model and computes mean ± std for MRR@10, Recall@1000.

Usage:
    python experiments/eval_multiseed_tot.py --model constbert --seed 42
    python experiments/eval_multiseed_tot.py --model colbert --seed 42
    python experiments/eval_multiseed_tot.py --aggregate
"""

import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.tot_loader import TRECToTDataLoader
from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import compute_mrr, compute_recall, compute_ndcg, compute_map

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_constbert_checkpoint(checkpoint_dir: Path, device: str):
    """Load fine-tuned ConstBERT model from checkpoint."""
    from transformers import AutoModel
    from safetensors.torch import load_file
    
    base_model = AutoModel.from_pretrained('pinecone/ConstBERT', trust_remote_code=True)
    
    safetensors_path = checkpoint_dir / "model.safetensors"
    pytorch_path = checkpoint_dir / "pytorch_model.bin"
    
    if safetensors_path.exists():
        logger.info(f"  Loading from {safetensors_path}")
        state_dict = load_file(str(safetensors_path))
        base_model.load_state_dict(state_dict, strict=False)
    elif pytorch_path.exists():
        logger.info(f"  Loading from {pytorch_path}")
        state_dict = torch.load(pytorch_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    model = ConstBERTWrapper.__new__(ConstBERTWrapper)
    model.model = base_model
    model.device = device
    model.model.to(device)
    model.model.eval()
    model.batch_size = 256
    model.tokenizer = base_model.query_tokenizer
    
    return model


def evaluate_constbert(seed: int, device: str = "cuda"):
    """Evaluate a single fine-tuned ConstBERT model on ToT TEST set."""
    
    logger.info("=" * 80)
    logger.info(f"Evaluating CONSTBERT seed={seed}")
    logger.info("=" * 80)
    
    checkpoint_dir = project_root / f"checkpoints/constbert_tot_seed{seed}"
    results_dir = project_root / "results" / "multiseed"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"constbert_seed{seed}_eval.json"
    
    if results_path.exists():
        logger.info(f"Results already exist at {results_path}, skipping...")
        with open(results_path) as f:
            return json.load(f)
    
    # Load dataset
    logger.info("\n[1/5] Loading ToT dataset...")
    loader = TRECToTDataLoader()
    stats = loader.get_corpus_statistics()
    
    doc_ids, passages = loader.load_corpus()
    logger.info(f"  Loaded {len(passages):,} documents")
    
    test_queries = loader.load_queries("test")
    test_qrels = loader.load_qrels("test")
    logger.info(f"  Loaded {len(test_queries)} test queries")
    
    # Load model
    logger.info(f"\n[2/5] Loading ConstBERT checkpoint (seed={seed})...")
    model = load_constbert_checkpoint(checkpoint_dir, device)
    logger.info("  Model loaded!")
    
    # Encode corpus
    logger.info("\n[3/5] Encoding corpus (this takes ~2 hours)...")
    start_time = time.time()
    embeddings = model.encode_documents(passages, batch_size=256, show_progress=True)
    encoding_time = time.time() - start_time
    logger.info(f"  Encoding completed in {encoding_time/60:.1f} minutes")
    
    # Build index
    logger.info("\n[4/5] Building FAISS index...")
    index = ConstBERTFAISSIndex(use_gpu=torch.cuda.is_available())
    index.add_documents(doc_ids, embeddings)
    logger.info(f"  Index contains {index.num_docs:,} documents")
    
    # Encode queries and retrieve
    logger.info("\n[5/5] Evaluating on TEST set...")
    query_ids = list(test_queries.keys())
    query_texts = [test_queries[qid] for qid in query_ids]
    query_embeddings = model.encode_queries(query_texts, batch_size=128, show_progress=True)
    
    all_results = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Retrieving")):
        doc_lists, score_lists = index.search(query_embeddings[i:i+1], k=1000, candidate_mult=10)
        all_results[qid] = {doc_id: float(score) for doc_id, score in zip(doc_lists[0], score_lists[0])}
    
    # Compute metrics
    mrr_10 = compute_mrr(all_results, test_qrels, k=10)
    recall_1000 = compute_recall(all_results, test_qrels, k=1000)
    ndcg_10 = compute_ndcg(all_results, test_qrels, k=10)
    
    logger.info(f"\n=== RESULTS for ConstBERT seed={seed} ===")
    logger.info(f"  MRR@10: {mrr_10*100:.2f}%")
    logger.info(f"  Recall@1000: {recall_1000*100:.2f}%")
    logger.info(f"  NDCG@10: {ndcg_10*100:.2f}%")
    
    results = {
        "model": "constbert",
        "seed": seed,
        "encoding_time_minutes": encoding_time / 60,
        "metrics": {
            "MRR@10": float(mrr_10),
            "Recall@1000": float(recall_1000),
            "NDCG@10": float(ndcg_10)
        }
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    return results


def aggregate_results():
    """Aggregate results from all seeds and compute mean ± std."""
    results_dir = project_root / "results" / "multiseed"
    seeds = [42, 123, 456]
    
    for model_type in ["constbert", "colbert"]:
        metrics_list = {"MRR@10": [], "Recall@1000": [], "NDCG@10": []}
        
        for seed in seeds:
            results_path = results_dir / f"{model_type}_seed{seed}_eval.json"
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                    for metric in metrics_list:
                        if metric in data["metrics"]:
                            metrics_list[metric].append(data["metrics"][metric])
        
        if metrics_list["MRR@10"]:
            logger.info(f"\n{model_type.upper()} (n={len(metrics_list['MRR@10'])}):")
            for metric, values in metrics_list.items():
                if values:
                    mean = np.mean(values) * 100
                    std = np.std(values) * 100
                    logger.info(f"  {metric}: {mean:.2f}% ± {std:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["constbert", "colbert"])
    parser.add_argument("--seed", type=int, choices=[42, 123, 456])
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    if args.aggregate:
        aggregate_results()
    elif args.model == "constbert" and args.seed:
        evaluate_constbert(args.seed, args.device)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()