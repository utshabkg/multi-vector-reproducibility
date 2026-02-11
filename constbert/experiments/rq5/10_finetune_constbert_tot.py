#!/usr/bin/env python3
"""
Fine-tune ConstBERT on TREC ToT 2025 dataset.
Adapted from ColBERT fine-tuning approach for ConstBERT architecture.
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import List, Tuple
import numpy as np
from datetime import datetime


class TOTTriplesDataset(Dataset):
    """Dataset for training triples (query, positive, negative)"""
    
    def __init__(self, triples_file: str, corpus_file: str, queries_file: str, max_samples=None, max_corpus_docs=None):
        self.triples = []
        self.queries = self._load_queries(queries_file)
        # Load triples first to know which docs we need
        needed_doc_ids = self._get_needed_doc_ids(triples_file, max_samples)
        self.corpus = self._load_corpus(corpus_file, max_corpus_docs, needed_doc_ids)
        self._load_triples(triples_file, max_samples)
    
    def _get_needed_doc_ids(self, triples_file, max_samples):
        """Get set of doc IDs referenced in triples"""
        doc_ids = set()
        with open(triples_file, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    _, pos_pid, neg_pid = parts
                    doc_ids.add(pos_pid)
                    doc_ids.add(neg_pid)
        return doc_ids
    
    def _load_corpus(self, corpus_file, max_docs=None, needed_doc_ids=None):
        """Load corpus documents"""
        print(f"Loading corpus from {corpus_file}...")
        corpus = {}
        with open(corpus_file, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Loading corpus")):
                if max_docs and len(corpus) >= max_docs and needed_doc_ids is None:
                    break
                doc = json.loads(line)
                doc_id = str(doc['id'])
                # If we have a list of needed docs, only load those
                if needed_doc_ids and doc_id not in needed_doc_ids:
                    continue
                text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                corpus[doc_id] = text
                # Stop if we've loaded all needed docs
                if needed_doc_ids and len(corpus) >= len(needed_doc_ids):
                    break
        print(f"Loaded {len(corpus):,} documents")
        return corpus
    
    def _load_queries(self, queries_file):
        """Load queries"""
        print(f"Loading queries from {queries_file}...")
        queries = {}
        with open(queries_file, 'r') as f:
            for line in f:
                q = json.loads(line)
                qid = str(q.get('query_id') or q.get('id'))
                text = q.get('query') or q.get('text')
                queries[qid] = text
        print(f"Loaded {len(queries):,} queries")
        return queries
    
    def _load_triples(self, triples_file, max_samples):
        """Load training triples"""
        print(f"Loading triples from {triples_file}...")
        with open(triples_file, 'r') as f:
            for i, line in enumerate(tqdm(f, desc="Loading triples")):
                if max_samples and i >= max_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue
                
                qid, pos_pid, neg_pid = parts
                
                query_text = self.queries.get(qid)
                pos_text = self.corpus.get(pos_pid)
                neg_text = self.corpus.get(neg_pid)
                
                if query_text and pos_text and neg_text:
                    self.triples.append((query_text, pos_text, neg_text))
        
        print(f"Loaded {len(self.triples):,} valid triples")
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, idx):
        return self.triples[idx]


class ConstBERTTrainer:
    """Trainer for ConstBERT fine-tuning"""
    
    def __init__(
        self,
        model_name: str = "pinecone/ConstBERT",
        device: str = "cuda",
        learning_rate: float = 5e-6,
        warmup_steps: int = 500,
        max_steps: int = 5000,
        batch_size: int = 8,  # Smaller due to memory constraints
        grad_accum_steps: int = 4,  # Effective batch size = 8*4=32
        output_dir: str = "checkpoints/constbert_tot_finetuned",
        log_steps: int = 50,
        save_steps: int = 1000,
        eval_steps: int = 500,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.output_dir = Path(output_dir)
        self.log_steps = log_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        
        print(f"\n{'='*80}")
        print(f"ConstBERT Fine-tuning Trainer")
        print(f"{'='*80}")
        print(f"Model: {model_name}")
        print(f"Device: {device}")
        print(f"Learning rate: {learning_rate}")
        print(f"Batch size: {batch_size} (effective: {batch_size * grad_accum_steps})")
        print(f"Max steps: {max_steps}")
        print(f"Output dir: {output_dir}")
        
        # Load model
        print(f"\nLoading ConstBERT model...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(device)
        
        # Set model to training mode
        self.model.train()
        
        print(f"✅ Model loaded successfully")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training log
        self.train_log = []
    
    def compute_contrastive_loss(
        self,
        query_embs: torch.Tensor,
        pos_embs: torch.Tensor,
        neg_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss with MaxSim scoring.
        
        Args:
            query_embs: (batch_size, num_query_tokens, dim)
            pos_embs: (batch_size, C=32, dim)
            neg_embs: (batch_size, C=32, dim)
        
        Returns:
            loss: scalar tensor
        """
        # Ensure float32 dtype for training
        query_embs = query_embs.float()
        pos_embs = pos_embs.float()
        neg_embs = neg_embs.float()
        
        # Normalize embeddings
        query_embs = F.normalize(query_embs, p=2, dim=-1)
        pos_embs = F.normalize(pos_embs, p=2, dim=-1)
        neg_embs = F.normalize(neg_embs, p=2, dim=-1)
        
        # Compute MaxSim scores for ConstBERT
        # ConstBERT: fixed C=32 vectors of dim=128
        # query_embs: (batch, C=32, dim=128)
        # pos_embs: (batch, C=32, dim=128)
        
        # Compute similarity matrix: (batch, 32, 32) - all query vectors vs all doc vectors
        pos_sim = torch.matmul(query_embs, pos_embs.transpose(1, 2))  # (batch, 32, 32)
        # MaxSim: for each query vector, take max similarity across all doc vectors, then sum
        pos_scores = pos_sim.max(dim=-1)[0].sum(dim=-1)  # (batch,)
        
        # Same for negatives
        neg_sim = torch.matmul(query_embs, neg_embs.transpose(1, 2))
        neg_scores = neg_sim.max(dim=-1)[0].sum(dim=-1)
        
        # Contrastive loss: margin = 1.0
        margin = 1.0
        loss = torch.relu(margin - pos_scores + neg_scores).mean()
        
        return loss, pos_scores.mean().item(), neg_scores.mean().item()
    
    def train_step(
        self,
        batch: Tuple[List[str], List[str], List[str]]
    ) -> Tuple[torch.Tensor, float, float]:
        """Single training step"""
        queries, positives, negatives = batch
        
        # Encode with ConstBERT (calling internal methods to preserve gradients)
        # The public encode_* methods have torch.no_grad(), so we call the internal _query/_doc methods
        
        # Tokenize queries
        query_input_ids, query_attention_mask = self.model.query_tokenizer.tensorize(queries)
        # Call internal _query method (preserves gradients)
        query_embs = self.model._query(query_input_ids, query_attention_mask)
        
        # Tokenize documents (positives)
        pos_input_ids, pos_attention_mask = self.model.doc_tokenizer.tensorize(positives)
        pos_embs = self.model._doc(pos_input_ids, pos_attention_mask)
        
        # Tokenize documents (negatives)
        neg_input_ids, neg_attention_mask = self.model.doc_tokenizer.tensorize(negatives)
        neg_embs = self.model._doc(neg_input_ids, neg_attention_mask)
        
        # Compute loss
        loss, pos_score, neg_score = self.compute_contrastive_loss(
            query_embs, pos_embs, neg_embs
        )
        
        return loss, pos_score, neg_score
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None
    ):
        """Main training loop"""
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = self.max_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"\n{'='*80}")
        print(f"Starting Training")
        print(f"{'='*80}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {self.warmup_steps}")
        print(f"Logging every: {self.log_steps} steps")
        print(f"Saving every: {self.save_steps} steps")
        
        # Training loop
        self.model.train()
        global_step = 0
        running_loss = 0.0
        running_pos_score = 0.0
        running_neg_score = 0.0
        optimizer.zero_grad()
        
        train_iterator = iter(train_loader)
        
        pbar = tqdm(total=total_steps, desc="Training")
        
        while global_step < total_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                batch = next(train_iterator)
            
            # Training step
            loss, pos_score, neg_score = self.train_step(batch)
            
            # Gradient accumulation
            loss = loss / self.grad_accum_steps
            loss.backward()
            
            running_loss += loss.item()
            running_pos_score += pos_score
            running_neg_score += neg_score
            
            # Update weights
            if (global_step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            pbar.update(1)
            
            # Logging
            if global_step % self.log_steps == 0:
                avg_loss = running_loss / self.log_steps
                avg_pos = running_pos_score / self.log_steps
                avg_neg = running_neg_score / self.log_steps
                lr = optimizer.param_groups[0]['lr']
                
                log_dict = {
                    'step': global_step,
                    'loss': avg_loss,
                    'pos_score': avg_pos,
                    'neg_score': avg_neg,
                    'margin': avg_pos - avg_neg,
                    'lr': lr,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.train_log.append(log_dict)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'pos': f'{avg_pos:.2f}',
                    'neg': f'{avg_neg:.2f}',
                    'margin': f'{avg_pos - avg_neg:.2f}'
                })
                
                running_loss = 0.0
                running_pos_score = 0.0
                running_neg_score = 0.0
            
            # Save checkpoint
            if global_step % self.save_steps == 0:
                self.save_checkpoint(global_step)
        
        pbar.close()
        
        # Save final model
        print("\n💾 Saving final model...")
        self.save_checkpoint(global_step, is_final=True)
        
        # Save training log
        log_file = self.output_dir / "training_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        
        print(f"✅ Training complete! Model saved to {self.output_dir}")
    
    def save_checkpoint(self, step: int, is_final: bool = False):
        """Save model checkpoint"""
        if is_final:
            checkpoint_dir = self.output_dir
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save metadata
        metadata = {
            'step': step,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_dir / 'training_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Checkpoint saved: {checkpoint_dir}")


def collate_fn(batch):
    """Collate function for DataLoader"""
    queries, positives, negatives = zip(*batch)
    return list(queries), list(positives), list(negatives)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune ConstBERT on ToT')
    parser.add_argument('--triples_file', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--corpus_file', type=str, required=True,
                       help='Path to corpus JSONL file')
    parser.add_argument('--queries_file', type=str, required=True,
                       help='Path to queries JSONL file')
    parser.add_argument('--output_dir', type=str, 
                       default='checkpoints/constbert_tot_finetuned',
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=5e-6,
                       help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=5000,
                       help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Warmup steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--test_mode', action='store_true',
                       help='Test mode with limited data')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"🎲 Random seed set: {args.seed}")
    
    # Load dataset
    max_samples = 100 if args.test_mode else None
    max_corpus_docs = 10000 if args.test_mode else None  # Only load 10K docs in test mode
    
    dataset = TOTTriplesDataset(
        triples_file=args.triples_file,
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        max_samples=max_samples,
        max_corpus_docs=max_corpus_docs
    )
    
    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize trainer
    trainer = ConstBERTTrainer(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps if not args.test_mode else 50,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        output_dir=args.output_dir
    )
    
    # Train
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
