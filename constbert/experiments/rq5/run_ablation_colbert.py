#!/usr/bin/env python3
"""
ColBERT Ablation Fine-tuning: Train on TRAIN+DEV1+DEV2, validate on DEV3, test on TEST.

This script must run in the colbertv2 conda environment!

Usage:
    conda activate colbertv2
    python experiments/run_ablation_colbert.py --test-mode
    python experiments/run_ablation_colbert.py --seed 42
    python experiments/run_ablation_colbert.py --full
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths for ColBERT imports - need to be in the colbert directory
COLBERT_SRC_DIR = Path(__file__).resolve().parent.parent.parent.parent / "colbertv2"
sys.path.insert(0, str(COLBERT_SRC_DIR))
sys.path.insert(0, str(COLBERT_SRC_DIR.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path("data/trec-tot-2025")
CORPUS_FILE = DATA_DIR / "trec-tot-2025-corpus.jsonl"
QUERIES_DIR = DATA_DIR / "queries"
QRELS_DIR = DATA_DIR / "qrel"

SEEDS = [42, 123, 456]

COLBERT_CONFIG = {
    "model_path": "colbert-ir/colbertv2.0",
    "max_steps": 1500,
    "batch_size": 16,
    "grad_accum": 2,
    "learning_rate": 3e-6,
    "warmup_steps": 200,
    "eval_steps": 100,
    "patience": 5,
    "embedding_dim": 128,
    "query_maxlen": 32,
    "doc_maxlen": 180,
}


# ============================================================================
# LOGGING
# ============================================================================

class Logger:
    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.start_time = time.time()
    
    def log(self, msg: str, level: str = "INFO"):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{elapsed:7.1f}s] [{self.prefix}] {level}: {msg}", flush=True)
    
    def info(self, msg: str): self.log(msg, "INFO")
    def success(self, msg: str): self.log(f"✅ {msg}", "SUCCESS")
    def error(self, msg: str): self.log(f"❌ {msg}", "ERROR")
    def progress(self, msg: str): self.log(f"⏳ {msg}", "PROGRESS")


# ============================================================================
# DATA LOADING
# ============================================================================

def load_queries(split: str) -> Dict[str, str]:
    query_file = QUERIES_DIR / f"{split}-2025-queries.jsonl"
    queries = {}
    with open(query_file) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q['query_id'])
            queries[qid] = q['query']
    return queries


def load_qrels(split: str) -> Dict[str, set]:
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
# DATASET
# ============================================================================

class TriplesDataset(Dataset):
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
                        self.triples.append((queries[qid], corpus[pos_id], corpus[neg_id]))
    
    def __len__(self): return len(self.triples)
    def __getitem__(self, idx): return self.triples[idx]


def collate_fn(batch):
    queries, positives, negatives = zip(*batch)
    return list(queries), list(positives), list(negatives)


# ============================================================================
# COLBERT TRAINER WITH EARLY STOPPING
# ============================================================================

class ColBERTTrainerWithEarlyStopping:
    def __init__(self, config: dict, output_dir: Path, seed: int, logger: Logger, test_mode: bool = False):
        self.config = config
        self.output_dir = output_dir
        self.seed = seed
        self.logger = logger
        self.test_mode = test_mode
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Set seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Import ColBERT model
        try:
            from src.colbert_model import ColBERT
        except ImportError:
            logger.error("Cannot import ColBERT. Make sure you're in colbertv2 environment!")
            raise
        
        # Load model
        self.logger.info(f"Loading ColBERT model from {config['model_path']}...")
        self.model = ColBERT(
            bert_model='bert-base-uncased',
            embedding_dim=config["embedding_dim"],
            query_maxlen=config["query_maxlen"],
            doc_maxlen=config["doc_maxlen"]
        )
        
        # Load pretrained weights
        model_path = Path(config["model_path"])
        pytorch_model = model_path / "pytorch_model.bin"
        if pytorch_model.exists():
            checkpoint = torch.load(pytorch_model, map_location='cpu')
            missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded ColBERT-v2 weights ({len(missing)} missing, {len(unexpected)} unexpected keys)")
        
        self.model = self.model.to(self.device)
        self.model.train()
        self.logger.success("Model loaded")
        
        # Training state
        self.train_log = []
        self.eval_log = []
        self.best_mrr = 0.0
        self.best_step = 0
        self.patience_counter = 0
        
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_loss(self, query_embs, pos_embs, neg_embs):
        """ColBERT MaxSim loss with margin.
        
        Note: ColBERT's query() and doc() methods already L2-normalize the embeddings.
        """
        # MaxSim scoring
        pos_sim = torch.matmul(query_embs, pos_embs.transpose(1, 2))
        pos_scores = pos_sim.max(dim=-1)[0].sum(dim=-1)
        
        neg_sim = torch.matmul(query_embs, neg_embs.transpose(1, 2))
        neg_scores = neg_sim.max(dim=-1)[0].sum(dim=-1)
        
        margin = 1.0
        loss = torch.relu(margin - pos_scores + neg_scores).mean()
        return loss, pos_scores.mean().item(), neg_scores.mean().item()
    
    def evaluate(self, val_queries: Dict[str, str], val_qrels: Dict[str, set], corpus: Dict[str, str]) -> float:
        """Quick MRR@10 evaluation with sampling."""
        self.model.eval()
        
        sample_queries = list(val_queries.items())[:10] if self.test_mode else list(val_queries.items())[:50]
        
        with torch.no_grad():
            mrrs = []
            for qid, query_text in sample_queries:
                if qid not in val_qrels:
                    continue
                
                relevant = val_qrels[qid]
                neg_docs = [d for d in list(corpus.keys())[:200] if d not in relevant]
                sample_docs = list(relevant) + neg_docs[:50]
                
                # Encode query using ColBERT's query() method
                q_emb = self.model.query([query_text[:256]])
                
                scores = []
                for doc_id in sample_docs:
                    if doc_id not in corpus:
                        continue
                    d_emb = self.model.doc([corpus[doc_id][:512]])
                    
                    # MaxSim scoring
                    sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                    score = sim.max(dim=-1)[0].sum().item()
                    scores.append((doc_id, score))
                
                scores.sort(key=lambda x: x[1], reverse=True)
                
                for rank, (doc_id, _) in enumerate(scores[:10], 1):
                    if doc_id in relevant:
                        mrrs.append(1.0 / rank)
                        break
                else:
                    mrrs.append(0.0)
        
        self.model.train()
        return np.mean(mrrs) if mrrs else 0.0
    
    def train(self, train_loader: DataLoader, val_queries: Dict[str, str], 
              val_qrels: Dict[str, set], corpus: Dict[str, str]):
        config = self.config
        max_steps = 50 if self.test_mode else config["max_steps"]
        eval_steps = 10 if self.test_mode else config["eval_steps"]
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=0.01
        )
        
        from transformers import get_linear_schedule_with_warmup
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
        
        pbar = tqdm(total=max_steps, desc=f"Training ColBERT (seed={self.seed})")
        
        while global_step < max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            queries, positives, negatives = batch
            
            # Forward pass using ColBERT's query() and doc() methods
            q_emb = self.model.query(queries)
            p_emb = self.model.doc(positives)
            n_emb = self.model.doc(negatives)
            
            loss, pos_score, neg_score = self.compute_loss(q_emb, p_emb, n_emb)
            
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
            pbar.set_postfix({"loss": f"{running_loss / global_step:.4f}"})
            
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
            
            self.train_log.append({
                "step": global_step,
                "loss": running_loss / global_step,
                "pos_score": pos_score,
                "neg_score": neg_score
            })
        
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
        torch.save(self.model.state_dict(), checkpoint_dir / "model_state.pt")
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump({"model_path": self.config["model_path"], "step": step, "seed": self.seed}, f)
        self.logger.info(f"Saved {'best ' if is_best else ''}checkpoint at step {step}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_on_test(model_path: Path, test_queries: Dict[str, str], test_qrels: Dict[str, set],
                     corpus: Dict[str, str], logger: Logger, test_mode: bool = False) -> Dict:
    """Evaluate fine-tuned ColBERT on TEST."""
    logger.info(f"Evaluating {model_path} on TEST...")
    
    device = "cuda"
    num_queries = 20 if test_mode else 100
    num_negs = 50 if test_mode else 100
    
    # Load model
    from src.colbert_model import ColBERT
    
    with open(model_path / "config.json") as f:
        config = json.load(f)
    
    model = ColBERT(
        bert_model='bert-base-uncased',
        embedding_dim=128,
        query_maxlen=32,
        doc_maxlen=180
    )
    state_dict = torch.load(model_path / "model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    # Sample queries
    eval_queries = [(qid, q) for qid, q in test_queries.items() if qid in test_qrels]
    random.seed(42)
    eval_queries = random.sample(eval_queries, min(num_queries, len(eval_queries)))
    
    logger.info(f"Evaluating {len(eval_queries)} sampled queries with {num_negs} negatives each...")
    
    mrrs = []
    
    with torch.no_grad():
        for qid, query_text in tqdm(eval_queries, desc="Evaluating"):
            relevant = test_qrels[qid]
            non_relevant = list(set(corpus.keys()) - relevant)
            sample_docs = list(relevant) + random.sample(non_relevant, min(num_negs, len(non_relevant)))
            
            # Use ColBERT's query() method (already normalized)
            q_emb = model.query([query_text[:256]])
            
            scores = []
            for doc_id in sample_docs:
                if doc_id not in corpus:
                    continue
                # Use ColBERT's doc() method (already normalized)
                d_emb = model.doc([corpus[doc_id][:512]])
                
                sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                score = sim.max(dim=-1)[0].sum().item()
                scores.append((doc_id, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, _) in enumerate(scores[:10], 1):
                if doc_id in relevant:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)
    
    return {"mrr@10": np.mean(mrrs), "num_queries": len(mrrs)}


def evaluate_pretrained_baseline(test_queries: Dict[str, str], test_qrels: Dict[str, set],
                                  corpus: Dict[str, str], logger: Logger, test_mode: bool = False) -> Dict:
    """Evaluate pretrained ColBERT-v2 as baseline."""
    logger.info("Evaluating pretrained ColBERT-v2 baseline...")
    
    device = "cuda"
    num_queries = 20 if test_mode else 100
    num_negs = 50 if test_mode else 100
    
    from src.colbert_model import ColBERT
    
    model = ColBERT(
        bert_model='bert-base-uncased',
        embedding_dim=128,
        query_maxlen=32,
        doc_maxlen=180
    )
    
    # Load pretrained weights
    model_path = Path(COLBERT_CONFIG["model_path"])
    pytorch_model = model_path / "pytorch_model.bin"
    if pytorch_model.exists():
        checkpoint = torch.load(pytorch_model, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    eval_queries = [(qid, q) for qid, q in test_queries.items() if qid in test_qrels]
    random.seed(42)
    eval_queries = random.sample(eval_queries, min(num_queries, len(eval_queries)))
    
    logger.info(f"Evaluating {len(eval_queries)} sampled queries...")
    
    mrrs = []
    
    with torch.no_grad():
        for qid, query_text in tqdm(eval_queries, desc="Evaluating baseline"):
            relevant = test_qrels[qid]
            non_relevant = list(set(corpus.keys()) - relevant)
            sample_docs = list(relevant) + random.sample(non_relevant, min(num_negs, len(non_relevant)))
            
            # Use ColBERT's query() method (already normalized)
            q_emb = model.query([query_text[:256]])
            
            scores = []
            for doc_id in sample_docs:
                if doc_id not in corpus:
                    continue
                # Use ColBERT's doc() method (already normalized)
                d_emb = model.doc([corpus[doc_id][:512]])
                
                sim = torch.matmul(q_emb, d_emb.transpose(1, 2))
                score = sim.max(dim=-1)[0].sum().item()
                scores.append((doc_id, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, _) in enumerate(scores[:10], 1):
                if doc_id in relevant:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)
    
    return {"mrr@10": np.mean(mrrs), "num_queries": len(mrrs)}


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_single_seed(seed: int, gpu_id: int, triples_file: Path, test_mode: bool = False):
    """Run ColBERT training for a single seed."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    logger = Logger(f"COLBERT-{seed}")
    logger.info(f"Starting on GPU {gpu_id}")
    
    output_dir = BASE_DIR / f"checkpoints/colbert_ablation_seed{seed}"
    
    # Load data
    logger.info("Loading training data...")
    train_queries = {}
    train_qrels = defaultdict(set)
    for split in ["train", "dev1", "dev2"]:
        train_queries.update(load_queries(split))
        for qid, docs in load_qrels(split).items():
            train_qrels[qid].update(docs)
    
    # Validation on DEV3
    val_queries = load_queries("dev3")
    val_qrels = load_qrels("dev3")
    
    # Test on TEST
    test_queries = load_queries("test")
    test_qrels = load_qrels("test")
    
    # Get all doc IDs we need
    all_doc_ids = set()
    for docs in train_qrels.values():
        all_doc_ids.update(docs)
    for docs in val_qrels.values():
        all_doc_ids.update(docs)
    for docs in test_qrels.values():
        all_doc_ids.update(docs)
    
    # Load triples to get negative doc IDs
    with open(triples_file) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                all_doc_ids.add(parts[1])
                all_doc_ids.add(parts[2])
    
    corpus = load_corpus_subset(all_doc_ids, logger)
    
    # Create dataset
    max_samples = 100 if test_mode else None
    dataset = TriplesDataset(triples_file, train_queries, corpus, max_samples)
    logger.info(f"Dataset size: {len(dataset)} triples")
    
    train_loader = DataLoader(
        dataset,
        batch_size=COLBERT_CONFIG["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Train
    trainer = ColBERTTrainerWithEarlyStopping(
        COLBERT_CONFIG,
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
    parser.add_argument("--test-mode", action="store_true", help="Quick test")
    parser.add_argument("--full", action="store_true", help="Full multi-seed experiment")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed")
    args = parser.parse_args()
    
    logger = Logger("MAIN")
    
    print("=" * 80)
    print("COLBERT ABLATION FINE-TUNING")
    print("Train on: TRAIN + DEV1 + DEV2 (428 queries)")
    print("Validate: DEV3 (536 queries) with early stopping")
    print("Test on: TEST (622 queries)")
    print("=" * 80)
    
    # Use same triples as ConstBERT
    triples_file = BASE_DIR / "data/tot/finetuning/combined_train_dev12.triples"
    
    if args.test_mode:
        triples_file = BASE_DIR / "data/tot/finetuning/test_mode.triples"
    
    if not triples_file.exists():
        logger.error(f"Triples file not found: {triples_file}")
        logger.error("Run ConstBERT ablation first to generate triples!")
        return
    
    if args.test_mode:
        logger.info("TEST MODE: Quick pipeline verification...")
        
        # Evaluate baseline
        test_queries = load_queries("test")
        test_qrels = load_qrels("test")
        
        all_doc_ids = set()
        for docs in test_qrels.values():
            all_doc_ids.update(docs)
        with open(CORPUS_FILE) as f:
            for i, line in enumerate(f):
                if i >= 10000:
                    break
                doc = json.loads(line)
                all_doc_ids.add(str(doc['id']))
        
        corpus = load_corpus_subset(all_doc_ids, logger)
        
        baseline_result = evaluate_pretrained_baseline(test_queries, test_qrels, corpus, logger, test_mode=True)
        
        # Train
        results = run_single_seed(42, 0, triples_file, test_mode=True)
        
        print("\n" + "=" * 80)
        print("TEST MODE COMPLETE")
        print("=" * 80)
        print(f"Pretrained baseline MRR@10: {baseline_result['mrr@10']:.4f}")
        print(f"Fine-tuned (428q) MRR@10:   {results['test_mrr@10']:.4f}")
        delta = results['test_mrr@10'] - baseline_result['mrr@10']
        print(f"Change: {'+' if delta >= 0 else ''}{delta:.4f} ({delta/baseline_result['mrr@10']*100:+.1f}%)")
        print("=" * 80)
        return
    
    if args.seed is not None:
        results = run_single_seed(args.seed, 0, triples_file, test_mode=False)
        print(f"\nResults for seed {args.seed}: {json.dumps(results, indent=2)}")
        return
    
    if args.full:
        logger.info(f"Running full experiment with seeds: {SEEDS}")
        
        # Evaluate baseline first
        test_queries = load_queries("test")
        test_qrels = load_qrels("test")
        
        all_doc_ids = set()
        for docs in test_qrels.values():
            all_doc_ids.update(docs)
        with open(CORPUS_FILE) as f:
            for i, line in enumerate(f):
                if len(all_doc_ids) >= 5000:
                    break
                doc = json.loads(line)
                all_doc_ids.add(str(doc['id']))
        
        corpus = load_corpus_subset(all_doc_ids, logger)
        
        baseline_result = evaluate_pretrained_baseline(test_queries, test_qrels, corpus, logger, test_mode=False)
        baseline_mrr = baseline_result['mrr@10']
        logger.success(f"Pretrained baseline MRR@10: {baseline_mrr:.4f}")
        
        # Run seeds sequentially (ColBERT is memory-intensive)
        all_results = []
        for seed in SEEDS:
            try:
                result = run_single_seed(seed, 0, triples_file, test_mode=False)
                all_results.append(result)
                logger.success(f"Seed {seed} complete: MRR={result['test_mrr@10']:.4f}")
            except Exception as e:
                logger.error(f"Seed {seed} failed: {e}")
        
        if all_results:
            mrrs = [r["test_mrr@10"] for r in all_results]
            mean_mrr = np.mean(mrrs)
            std_mrr = np.std(mrrs)
            delta = mean_mrr - baseline_mrr
            
            print("\n" + "=" * 80)
            print("COLBERT ABLATION RESULTS")
            print("=" * 80)
            print(f"\n📊 COMPARISON:")
            print(f"   Pretrained baseline:     {baseline_mrr:.4f}")
            print(f"   Fine-tuned (428q):       {mean_mrr:.4f} ± {std_mrr:.4f}")
            print(f"   Change:                  {'+' if delta >= 0 else ''}{delta:.4f} ({delta/baseline_mrr*100:+.1f}%)")
            print(f"\n📈 INDIVIDUAL SEEDS:")
            for r in all_results:
                print(f"   Seed {r['seed']}: MRR@10={r['test_mrr@10']:.4f} (best DEV3: {r['best_dev3_mrr']:.4f} at step {r['best_step']})")
            print("=" * 80)
            
            summary = {
                "experiment": "colbert_ablation_finetuning",
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
            
            summary_file = BASE_DIR / "results/colbert_ablation_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"\nSummary saved to: {summary_file}")
    
    else:
        print("\nUsage:")
        print("  --test-mode    Quick pipeline test")
        print("  --full         Full multi-seed experiment")
        print("  --seed 42      Run single seed")


if __name__ == "__main__":
    main()
