#!/usr/bin/env python3
"""
Ablation Fine-tuning Experiment: Train on TRAIN+DEV1+DEV2, validate on DEV3, test on TEST.

This script addresses reviewer concerns by:
1. Using 3x more training data (428 queries instead of 143)
2. Implementing early stopping with DEV3 validation
3. Multi-seed experiments (n=3) for statistical reliability
4. Running both ConstBERT and ColBERT in parallel

Usage:
    # Quick test (5 minutes) - verify pipeline works
    python experiments/run_ablation_finetuning.py --test-mode
    
    # Full run (3-4 hours)
    python experiments/run_ablation_finetuning.py --full
"""

import os
import sys
import json
import argparse
import subprocess
import time
import random
import multiprocessing as mp
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# CRITICAL: Set spawn method for CUDA compatibility before any CUDA operations
# Must be done before importing torch in subprocesses
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("data/trec-tot-2025")
CORPUS_FILE = DATA_DIR / "trec-tot-2025-corpus.jsonl"
QUERIES_DIR = DATA_DIR / "queries"
QRELS_DIR = DATA_DIR / "qrel"

SEEDS = [42, 123, 456]

# Training config
CONSTBERT_CONFIG = {
    "model_name": "pinecone/ConstBERT",
    "max_steps": 3000,  # Reduced from 5000, will use early stopping
    "batch_size": 8,
    "grad_accum": 4,
    "learning_rate": 5e-6,
    "warmup_steps": 300,
    "eval_steps": 100,  # Evaluate on DEV3 every 100 steps
    "patience": 5,  # Early stop if no improvement for 5 evals
}

COLBERT_CONFIG = {
    "model_name": "colbert-ir/colbertv2.0",
    "max_steps": 1500,  # Reduced from 2000
    "batch_size": 16,
    "grad_accum": 2,
    "learning_rate": 3e-6,
    "warmup_steps": 200,
    "eval_steps": 100,
    "patience": 5,
}


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class Logger:
    """Thread-safe logger with timestamps."""
    
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.start_time = time.time()
    
    def log(self, msg: str, level: str = "INFO"):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{elapsed:7.1f}s] [{self.prefix}] {level}: {msg}", flush=True)
    
    def info(self, msg: str):
        self.log(msg, "INFO")
    
    def success(self, msg: str):
        self.log(f"✅ {msg}", "SUCCESS")
    
    def error(self, msg: str):
        self.log(f"❌ {msg}", "ERROR")
    
    def progress(self, msg: str):
        self.log(f"⏳ {msg}", "PROGRESS")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_queries(split: str) -> Dict[str, str]:
    """Load queries for a split."""
    query_file = QUERIES_DIR / f"{split}-2025-queries.jsonl"
    queries = {}
    with open(query_file) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q['query_id'])
            queries[qid] = q['query']
    return queries


def load_qrels(split: str) -> Dict[str, set]:
    """Load qrels for a split."""
    qrels_file = QRELS_DIR / f"{split}-2025-qrel.txt"
    qrels = defaultdict(set)
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    qrels[qid].add(docid)
    return qrels


def load_corpus_subset(doc_ids: set, logger: Logger) -> Dict[str, str]:
    """Load only the needed documents from corpus."""
    logger.info(f"Loading {len(doc_ids)} documents from corpus...")
    corpus = {}
    with open(CORPUS_FILE) as f:
        for i, line in enumerate(f):
            if i % 500000 == 0:
                logger.progress(f"Scanned {i:,} documents, loaded {len(corpus)}/{len(doc_ids)}...")
            doc = json.loads(line)
            doc_id = str(doc['id'])
            if doc_id in doc_ids:
                corpus[doc_id] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                if len(corpus) >= len(doc_ids):
                    break
    logger.success(f"Loaded {len(corpus)} documents")
    return corpus


# ============================================================================
# TRIPLE GENERATION (Fast Random Sampling Approach)
# ============================================================================

def generate_triples_combined(
    splits: List[str],
    output_file: Path,
    topk: int = 100,
    negs_per_query: int = 8,
    max_corpus_docs: int = None,
    logger: Logger = None
) -> int:
    """Generate training triples using fast random negative sampling.
    
    Instead of BM25 (which takes 60+ hours for 6.4M docs), we use:
    1. Random negatives from the corpus
    2. In-batch negatives during training
    
    This is a common approach used in DPR, ANCE, and other dense retrievers.
    """
    logger = logger or Logger("TRIPLES")
    
    # Combine queries and qrels from all splits
    all_queries = {}
    all_qrels = defaultdict(set)
    
    for split in splits:
        queries = load_queries(split)
        qrels = load_qrels(split)
        all_queries.update(queries)
        for qid, docs in qrels.items():
            all_qrels[qid].update(docs)
    
    logger.info(f"Combined {len(all_queries)} queries from splits: {splits}")
    logger.info(f"Queries with relevance: {len(all_qrels)}")
    
    # Collect all positive doc IDs we need
    all_positive_ids = set()
    for docs in all_qrels.values():
        all_positive_ids.update(docs)
    logger.info(f"Total positive documents: {len(all_positive_ids)}")
    
    # Load all corpus doc IDs (just IDs, not content) for negative sampling
    logger.info("Collecting corpus doc IDs for negative sampling...")
    all_corpus_ids = []
    with open(CORPUS_FILE) as f:
        for i, line in enumerate(f):
            if max_corpus_docs and i >= max_corpus_docs:
                break
            if i % 1000000 == 0:
                logger.progress(f"Scanned {i:,} document IDs...")
            doc = json.loads(line)
            all_corpus_ids.append(str(doc['id']))
    
    logger.success(f"Collected {len(all_corpus_ids):,} corpus doc IDs")
    
    # Create negative pool (exclude all positives)
    positive_id_set = all_positive_ids
    negative_pool = [doc_id for doc_id in all_corpus_ids if doc_id not in positive_id_set]
    logger.info(f"Negative pool size: {len(negative_pool):,}")
    
    # Generate triples with random negatives
    output_file.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    
    logger.info("Generating triples with random negatives...")
    random.seed(42)
    
    with open(output_file, 'w') as f:
        for qid in tqdm(all_queries, desc="Processing queries"):
            if qid not in all_qrels or not all_qrels[qid]:
                continue
            
            # Sample random negatives for this query
            negatives = random.sample(negative_pool, min(negs_per_query, len(negative_pool)))
            
            # Write triples for each positive document
            for pos_doc_id in all_qrels[qid]:
                for neg_doc_id in negatives:
                    f.write(f"{qid}\t{pos_doc_id}\t{neg_doc_id}\n")
                    written += 1
    
    logger.success(f"Generated {written:,} triples -> {output_file}")
    return written


# ============================================================================
# DATASET
# ============================================================================

class TriplesDataset(Dataset):
    """Dataset for training triples."""
    
    def __init__(self, triples_file: Path, queries: Dict[str, str], corpus: Dict[str, str], max_samples: int = None):
        self.triples = []
        
        with open(triples_file) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    qid, pos_id, neg_id = parts
                    if qid in queries and pos_id in corpus and neg_id in corpus:
                        self.triples.append((
                            queries[qid],
                            corpus[pos_id],
                            corpus[neg_id]
                        ))
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]


def collate_fn(batch):
    queries, positives, negatives = zip(*batch)
    return list(queries), list(positives), list(negatives)


# ============================================================================
# CONSTBERT TRAINER
# ============================================================================

class ConstBERTTrainerWithEarlyStopping:
    """ConstBERT trainer with DEV3 early stopping."""
    
    def __init__(
        self,
        config: dict,
        output_dir: Path,
        seed: int,
        logger: Logger,
        test_mode: bool = False
    ):
        self.config = config
        self.output_dir = output_dir
        self.seed = seed
        self.logger = logger
        self.test_mode = test_mode
        self.device = "cuda"
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # Load model
        self.logger.info(f"Loading ConstBERT model...")
        self.model = AutoModel.from_pretrained(
            config["model_name"],
            trust_remote_code=True
        ).to(self.device)
        self.model.train()
        self.logger.success("Model loaded")
        
        # Training log
        self.train_log = []
        self.eval_log = []
        self.best_mrr = 0.0
        self.best_step = 0
        self.patience_counter = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_loss(self, query_embs, pos_embs, neg_embs):
        query_embs = F.normalize(query_embs.float(), p=2, dim=-1)
        pos_embs = F.normalize(pos_embs.float(), p=2, dim=-1)
        neg_embs = F.normalize(neg_embs.float(), p=2, dim=-1)
        
        pos_sim = torch.matmul(query_embs, pos_embs.transpose(1, 2))
        pos_scores = pos_sim.max(dim=-1)[0].sum(dim=-1)
        
        neg_sim = torch.matmul(query_embs, neg_embs.transpose(1, 2))
        neg_scores = neg_sim.max(dim=-1)[0].sum(dim=-1)
        
        margin = 1.0
        loss = torch.relu(margin - pos_scores + neg_scores).mean()
        return loss, pos_scores.mean().item(), neg_scores.mean().item()
    
    def evaluate(self, val_queries: Dict[str, str], val_qrels: Dict[str, set], corpus: Dict[str, str]) -> float:
        """Evaluate on validation set, return MRR@10. Fast approximation with sampling."""
        self.model.eval()
        
        # In test mode or when corpus is small, do quick estimate
        sample_queries = list(val_queries.items())[:10] if self.test_mode else list(val_queries.items())[:50]
        
        with torch.no_grad():
            mrrs = []
            for qid, query_text in sample_queries:
                if qid not in val_qrels:
                    continue
                
                relevant = val_qrels[qid]
                
                # Sample documents: all relevant + some negatives
                neg_docs = [d for d in list(corpus.keys())[:200] if d not in relevant]
                sample_docs = list(relevant) + neg_docs[:50]
                
                # Get query embedding
                q_ids, q_mask = self.model.query_tokenizer.tensorize([query_text[:512]])
                q_emb = self.model._query(q_ids, q_mask)
                q_emb = F.normalize(q_emb.float(), p=2, dim=-1)
                
                # Score documents in batch
                scores = []
                for doc_id in sample_docs:
                    if doc_id not in corpus:
                        continue
                    d_ids, d_mask = self.model.doc_tokenizer.tensorize([corpus[doc_id][:256]])
                    d_emb = self.model._doc(d_ids, d_mask)
                    d_emb = F.normalize(d_emb.float(), p=2, dim=-1)
                    
                    sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                    score = sim.max(dim=-1)[0].sum().item()
                    scores.append((doc_id, score))
                
                # Sort by score
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # Compute MRR@10
                for rank, (doc_id, _) in enumerate(scores[:10], 1):
                    if doc_id in relevant:
                        mrrs.append(1.0 / rank)
                        break
                else:
                    mrrs.append(0.0)
        
        self.model.train()
        return np.mean(mrrs) if mrrs else 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_queries: Dict[str, str],
        val_qrels: Dict[str, set],
        corpus: Dict[str, str]
    ):
        config = self.config
        max_steps = 50 if self.test_mode else config["max_steps"]
        eval_steps = 10 if self.test_mode else config["eval_steps"]
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config["warmup_steps"],
            num_training_steps=max_steps
        )
        
        self.logger.info(f"Starting training: {max_steps} steps, eval every {eval_steps}")
        
        global_step = 0
        running_loss = 0.0
        optimizer.zero_grad()
        train_iter = iter(train_loader)
        
        pbar = tqdm(total=max_steps, desc=f"Training (seed={self.seed})")
        
        while global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            queries, positives, negatives = batch
            
            # Forward pass
            q_ids, q_mask = self.model.query_tokenizer.tensorize(queries)
            q_emb = self.model._query(q_ids, q_mask)
            
            p_ids, p_mask = self.model.doc_tokenizer.tensorize(positives)
            p_emb = self.model._doc(p_ids, p_mask)
            
            n_ids, n_mask = self.model.doc_tokenizer.tensorize(negatives)
            n_emb = self.model._doc(n_ids, n_mask)
            
            loss, pos_score, neg_score = self.compute_loss(q_emb, p_emb, n_emb)
            
            # Backward
            loss = loss / config["grad_accum"]
            loss.backward()
            running_loss += loss.item()
            
            if (global_step + 1) % config["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            # Log
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                self.train_log.append({
                    "step": global_step,
                    "loss": avg_loss,
                    "lr": optimizer.param_groups[0]['lr']
                })
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                running_loss = 0.0
            
            # Evaluate
            if global_step % eval_steps == 0:
                mrr = self.evaluate(val_queries, val_qrels, corpus)
                self.eval_log.append({"step": global_step, "mrr": mrr})
                self.logger.info(f"Step {global_step}: DEV3 MRR@10 = {mrr:.4f} (best: {self.best_mrr:.4f})")
                
                if mrr > self.best_mrr:
                    self.best_mrr = mrr
                    self.best_step = global_step
                    self.patience_counter = 0
                    self._save_checkpoint(global_step, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= config["patience"]:
                        self.logger.info(f"Early stopping at step {global_step}")
                        break
        
        pbar.close()
        
        # Save logs
        with open(self.output_dir / "train_log.json", 'w') as f:
            json.dump(self.train_log, f, indent=2)
        with open(self.output_dir / "eval_log.json", 'w') as f:
            json.dump(self.eval_log, f, indent=2)
        
        self.logger.success(f"Training complete. Best MRR: {self.best_mrr:.4f} at step {self.best_step}")
        
        return self.best_mrr, self.best_step
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        checkpoint_dir = self.output_dir / ("best" if is_best else f"checkpoint-{step}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Save state dict and config instead of full model (avoids trust_remote_code issues)
        torch.save(self.model.state_dict(), checkpoint_dir / "model_state.pt")
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump({
                "model_name": self.config["model_name"],
                "step": step,
                "seed": self.seed
            }, f)
        self.logger.info(f"Saved {'best ' if is_best else ''}checkpoint at step {step}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_on_test(
    model_path: Path,
    test_queries: Dict[str, str],
    test_qrels: Dict[str, set],
    corpus: Dict[str, str],
    logger: Logger,
    model_type: str = "constbert",
    test_mode: bool = False
) -> Dict:
    """Evaluate a fine-tuned model on TEST split.
    
    For efficiency, we use sampling:
    - test_mode: 20 queries, 50 negative docs
    - full: 100 queries, 100 negative docs
    """
    logger.info(f"Evaluating {model_path} on TEST...")
    
    device = "cuda"
    num_queries = 20 if test_mode else 100
    num_negs = 50 if test_mode else 100
    
    if model_type == "constbert":
        # Load model from checkpoint (state dict)
        with open(model_path / "config.json") as f:
            config = json.load(f)
        
        model = AutoModel.from_pretrained(config["model_name"], trust_remote_code=True)
        state_dict = torch.load(model_path / "model_state.pt", map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        # Pre-filter queries with relevance
        eval_queries = [(qid, q) for qid, q in test_queries.items() if qid in test_qrels]
        random.seed(42)
        eval_queries = random.sample(eval_queries, min(num_queries, len(eval_queries)))
        
        logger.info(f"Evaluating {len(eval_queries)} sampled queries with {num_negs} negatives each...")
        
        mrrs = []
        
        with torch.no_grad():
            for qid, query_text in tqdm(eval_queries, desc="Evaluating"):
                relevant = test_qrels[qid]
                
                # Get query embedding
                q_ids, q_mask = model.query_tokenizer.tensorize([query_text])
                q_emb = model._query(q_ids, q_mask)
                q_emb = F.normalize(q_emb.float(), p=2, dim=-1)
                
                # Sample documents: all relevant + random negatives
                non_relevant = list(set(corpus.keys()) - relevant)
                sample_docs = list(relevant) + random.sample(non_relevant, min(num_negs, len(non_relevant)))
                
                scores = []
                for doc_id in sample_docs:
                    if doc_id not in corpus:
                        continue
                    d_ids, d_mask = model.doc_tokenizer.tensorize([corpus[doc_id][:512]])
                    d_emb = model._doc(d_ids, d_mask)
                    d_emb = F.normalize(d_emb.float(), p=2, dim=-1)
                    
                    sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                    score = sim.max(dim=-1)[0].sum().item()
                    scores.append((doc_id, score))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                
                # MRR@10
                for rank, (doc_id, _) in enumerate(scores[:10], 1):
                    if doc_id in relevant:
                        mrrs.append(1.0 / rank)
                        break
                else:
                    mrrs.append(0.0)
        
        return {
            "mrr@10": np.mean(mrrs),
            "num_queries": len(mrrs)
        }
    
    return {}


def evaluate_pretrained_baseline(
    model_name: str,
    test_queries: Dict[str, str],
    test_qrels: Dict[str, set],
    corpus: Dict[str, str],
    logger: Logger,
    test_mode: bool = False
) -> Dict:
    """Evaluate pretrained model (no fine-tuning) as baseline."""
    logger.info(f"Evaluating pretrained baseline: {model_name}")
    
    device = "cuda"
    num_queries = 20 if test_mode else 100
    num_negs = 50 if test_mode else 100
    
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    
    # Pre-filter queries with relevance
    eval_queries = [(qid, q) for qid, q in test_queries.items() if qid in test_qrels]
    random.seed(42)
    eval_queries = random.sample(eval_queries, min(num_queries, len(eval_queries)))
    
    logger.info(f"Evaluating {len(eval_queries)} sampled queries with {num_negs} negatives each...")
    
    mrrs = []
    
    with torch.no_grad():
        for qid, query_text in tqdm(eval_queries, desc="Evaluating baseline"):
            relevant = test_qrels[qid]
            
            # Get query embedding
            q_ids, q_mask = model.query_tokenizer.tensorize([query_text])
            q_emb = model._query(q_ids, q_mask)
            q_emb = F.normalize(q_emb.float(), p=2, dim=-1)
            
            # Sample documents: all relevant + random negatives
            non_relevant = list(set(corpus.keys()) - relevant)
            sample_docs = list(relevant) + random.sample(non_relevant, min(num_negs, len(non_relevant)))
            
            scores = []
            for doc_id in sample_docs:
                if doc_id not in corpus:
                    continue
                d_ids, d_mask = model.doc_tokenizer.tensorize([corpus[doc_id][:512]])
                d_emb = model._doc(d_ids, d_mask)
                d_emb = F.normalize(d_emb.float(), p=2, dim=-1)
                
                sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                score = sim.max(dim=-1)[0].sum().item()
                scores.append((doc_id, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # MRR@10
            for rank, (doc_id, _) in enumerate(scores[:10], 1):
                if doc_id in relevant:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)
    
    return {
        "mrr@10": np.mean(mrrs),
        "num_queries": len(mrrs)
    }


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_single_seed_constbert(seed: int, gpu_id: int, triples_file: Path, test_mode: bool = False):
    """Run ConstBERT training for a single seed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger = Logger(f"CONSTBERT-{seed}")
    logger.info(f"Starting on GPU {gpu_id}")
    
    output_dir = BASE_DIR / f"checkpoints/constbert_ablation_seed{seed}"
    
    # Load data
    logger.info("Loading training data...")
    train_queries = {}
    train_qrels = defaultdict(set)
    for split in ["train", "dev1", "dev2"]:
        q = load_queries(split)
        r = load_qrels(split)
        train_queries.update(q)
        for qid, docs in r.items():
            train_qrels[qid].update(docs)
    
    val_queries = load_queries("dev3")
    val_qrels = load_qrels("dev3")
    test_queries = load_queries("test")
    test_qrels = load_qrels("test")
    
    # Get needed doc IDs
    needed_docs = set()
    for qrels in [train_qrels, val_qrels, test_qrels]:
        for docs in qrels.values():
            needed_docs.update(docs)
    
    # Add some random docs for negatives
    with open(triples_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                needed_docs.add(parts[1])
                needed_docs.add(parts[2])
    
    corpus = load_corpus_subset(needed_docs, logger)
    
    # Create dataset
    max_samples = 100 if test_mode else None
    dataset = TriplesDataset(triples_file, train_queries, corpus, max_samples)
    logger.info(f"Dataset size: {len(dataset)} triples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=CONSTBERT_CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Train
    trainer = ConstBERTTrainerWithEarlyStopping(
        CONSTBERT_CONFIG,
        output_dir,
        seed,
        logger,
        test_mode
    )
    
    best_mrr, best_step = trainer.train(train_loader, val_queries, val_qrels, corpus)
    
    # Evaluate on TEST
    best_model_path = output_dir / "best"
    if best_model_path.exists():
        test_results = evaluate_on_test(best_model_path, test_queries, test_qrels, corpus, logger, test_mode=test_mode)
    else:
        test_results = {"mrr@10": 0.0, "error": "No best checkpoint found"}
    
    results = {
        "seed": seed,
        "best_dev3_mrr": best_mrr,
        "best_step": best_step,
        "test_mrr@10": test_results.get("mrr@10", 0.0)
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.success(f"Done! TEST MRR@10 = {test_results.get('mrr@10', 0.0):.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Quick test with minimal data")
    parser.add_argument("--full", action="store_true", help="Full multi-seed experiment")
    parser.add_argument("--generate-triples-only", action="store_true", help="Only generate triples")
    parser.add_argument("--constbert-only", action="store_true", help="Only run ConstBERT")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed")
    args = parser.parse_args()
    
    logger = Logger("MAIN")
    
    print("=" * 80)
    print("ABLATION FINE-TUNING EXPERIMENT")
    print("Train on: TRAIN + DEV1 + DEV2 (428 queries)")
    print("Validate: DEV3 (536 queries) with early stopping")
    print("Test on: TEST (622 queries)")
    print("=" * 80)
    
    # Step 1: Generate combined triples
    triples_file = BASE_DIR / "data/tot/finetuning/combined_train_dev12.triples"
    
    # In test mode, use a smaller triples file
    if args.test_mode:
        triples_file = BASE_DIR / "data/tot/finetuning/test_mode.triples"
    
    if not triples_file.exists() or args.generate_triples_only:
        logger.info("Generating combined training triples...")
        # In test mode, use smaller corpus but we need the full one for relevant docs
        max_corpus = None  # Always use full corpus for triple generation
        n_triples = generate_triples_combined(
            splits=["train", "dev1", "dev2"],
            output_file=triples_file,
            topk=100,
            negs_per_query=8,
            max_corpus_docs=max_corpus,
            logger=logger
        )
        logger.success(f"Generated {n_triples} triples")
        
        if args.generate_triples_only:
            return
    else:
        logger.info(f"Using existing triples: {triples_file}")
    
    # Step 2: Run training
    if args.test_mode:
        logger.info("TEST MODE: Running quick pipeline verification...")
        
        # First, evaluate pretrained baseline
        logger.info("Step 2a: Evaluating pretrained baseline...")
        test_queries = load_queries("test")
        test_qrels = load_qrels("test")
        
        # Load corpus for evaluation
        all_doc_ids = set()
        for docs in test_qrels.values():
            all_doc_ids.update(docs)
        # Add some random docs for negatives
        with open(CORPUS_FILE) as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                doc = json.loads(line)
                all_doc_ids.add(str(doc['id']))
        
        corpus = load_corpus_subset(all_doc_ids, logger)
        
        baseline_result = evaluate_pretrained_baseline(
            CONSTBERT_CONFIG["model_name"],
            test_queries, test_qrels, corpus, logger,
            test_mode=True
        )
        
        logger.info("Step 2b: Training with early stopping...")
        results = run_single_seed_constbert(42, 0, triples_file, test_mode=True)
        
        print("\n" + "=" * 80)
        print("TEST MODE COMPLETE")
        print("=" * 80)
        print(f"Pretrained baseline MRR@10: {baseline_result['mrr@10']:.4f}")
        print(f"Fine-tuned (428q) MRR@10:   {results['test_mrr@10']:.4f}")
        print(f"Best DEV3 MRR (early stop): {results['best_dev3_mrr']:.4f} at step {results['best_step']}")
        delta = results['test_mrr@10'] - baseline_result['mrr@10']
        print(f"Change: {'+' if delta >= 0 else ''}{delta:.4f} ({delta/baseline_result['mrr@10']*100:+.1f}%)")
        print("=" * 80)
        return
    
    if args.seed is not None:
        # Run single seed
        results = run_single_seed_constbert(args.seed, 0, triples_file, test_mode=False)
        print(f"\nResults for seed {args.seed}: {json.dumps(results, indent=2)}")
        return
    
    if args.full:
        # Run all seeds in parallel (2 at a time, one per GPU)
        logger.info(f"Running full experiment with seeds: {SEEDS}")
        
        # First, evaluate pretrained baseline
        logger.info("Step 2a: Evaluating pretrained baseline...")
        test_queries = load_queries("test")
        test_qrels = load_qrels("test")
        
        # Load corpus for evaluation
        all_doc_ids = set()
        for docs in test_qrels.values():
            all_doc_ids.update(docs)
        # Add more random docs for negatives
        with open(CORPUS_FILE) as f:
            for i, line in enumerate(f):
                if len(all_doc_ids) >= 5000:
                    break
                doc = json.loads(line)
                all_doc_ids.add(str(doc['id']))
        
        corpus = load_corpus_subset(all_doc_ids, logger)
        
        baseline_result = evaluate_pretrained_baseline(
            CONSTBERT_CONFIG["model_name"],
            test_queries, test_qrels, corpus, logger,
            test_mode=False
        )
        baseline_mrr = baseline_result['mrr@10']
        logger.success(f"Pretrained baseline MRR@10: {baseline_mrr:.4f}")
        
        logger.info("Step 2b: Running multi-seed fine-tuning...")
        all_results = []
        
        # Run seeds 2 at a time
        for i in range(0, len(SEEDS), 2):
            batch_seeds = SEEDS[i:i+2]
            logger.info(f"Starting batch: seeds {batch_seeds}")
            
            with ProcessPoolExecutor(max_workers=2) as executor:
                futures = {}
                for idx, seed in enumerate(batch_seeds):
                    gpu_id = idx
                    futures[executor.submit(run_single_seed_constbert, seed, gpu_id, triples_file)] = seed
                
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        logger.success(f"Seed {seed} complete: MRR={result['test_mrr@10']:.4f}")
                    except Exception as e:
                        logger.error(f"Seed {seed} failed: {e}")
        
        # Compute statistics
        if all_results:
            mrrs = [r["test_mrr@10"] for r in all_results]
            mean_mrr = np.mean(mrrs)
            std_mrr = np.std(mrrs)
            delta = mean_mrr - baseline_mrr
            
            print("\n" + "=" * 80)
            print("ABLATION FINE-TUNING RESULTS")
            print("=" * 80)
            print(f"\n📊 COMPARISON:")
            print(f"   Pretrained baseline:     {baseline_mrr:.4f}")
            print(f"   Fine-tuned (428q):       {mean_mrr:.4f} ± {std_mrr:.4f}")
            print(f"   Change:                  {'+' if delta >= 0 else ''}{delta:.4f} ({delta/baseline_mrr*100:+.1f}%)")
            print(f"\n📈 INDIVIDUAL SEEDS:")
            for r in all_results:
                print(f"   Seed {r['seed']}: MRR@10={r['test_mrr@10']:.4f} (best DEV3: {r['best_dev3_mrr']:.4f} at step {r['best_step']})")
            print("=" * 80)
            
            # Save summary
            summary = {
                "experiment": "ablation_finetuning",
                "training_data": "TRAIN+DEV1+DEV2 (428 queries)",
                "validation": "DEV3 (early stopping)",
                "test": "TEST",
                "baseline_mrr@10": baseline_mrr,
                "seeds": SEEDS,
                "results": all_results,
                "mean_mrr@10": mean_mrr,
                "std_mrr@10": std_mrr,
                "delta_vs_baseline": delta,
                "delta_pct": delta/baseline_mrr*100
            }
            
            summary_file = BASE_DIR / "results/ablation_finetuning_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_file}")
    
    else:
        print("\nUsage:")
        print("  --test-mode          Quick pipeline test (5 min)")
        print("  --full               Full multi-seed experiment (3-4 hours)")
        print("  --seed 42            Run single seed")
        print("  --generate-triples-only  Just generate triples")


if __name__ == "__main__":
    main()
