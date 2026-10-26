#!/usr/bin/env python3
"""
ColBERT Training Implementation
Implements pairwise softmax loss with in-batch negatives as described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from pathlib import Path
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import logging

try:
    from .colbert_model import ColBERT
except ImportError:
    from colbert_model import ColBERT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColBERTTrainer:
    """
    Trainer for ColBERT model with pairwise softmax cross-entropy loss.

    Implements:
    - In-batch negatives: All other positives in batch serve as negatives
    - Hard negative mining: Optional BM25-based hard negatives
    - Gradient accumulation for large effective batch sizes
    - Checkpointing and logging
    """

    def __init__(
        self,
        model: ColBERT,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 3e-6,
        warmup_steps: int = 1000,
        accumulation_steps: int = 1,
        max_grad_norm: float = 2.0,
        weight_decay: float = 0.01,
        checkpoint_dir: str = "models/checkpoints",
        log_every: int = 10,
        eval_every: int = 500,
        save_every: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.

        Args:
            model: ColBERT model to train
            train_dataloader: DataLoader for training data
            val_dataloader: Optional DataLoader for validation
            learning_rate: Learning rate (paper uses 3e-6)
            warmup_steps: Linear warmup steps
            accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            weight_decay: Weight decay for AdamW optimizer
            checkpoint_dir: Directory to save checkpoints
            log_every: Log training metrics every N steps
            eval_every: Evaluate on validation set every N steps
            save_every: Save checkpoint every N steps
            device: Device to train on
        """
        self.model = model.to(device)

        # Enable DataParallel for multi-GPU training
        if torch.cuda.device_count() > 1:
            logger.info(
                f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
            self.multi_gpu = True
        else:
            self.multi_gpu = False

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.weight_decay = weight_decay

        # Logging and checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every

        # Optimizer (AdamW as in paper)
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (linear warmup)
        # Only enable if warmup_steps > 0 (author uses constant LR, no warmup)
        if warmup_steps > 0:
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-3,  # Start at 0.001 * lr
                end_factor=1.0,     # End at full lr
                total_iters=warmup_steps
            )
        else:
            self.scheduler = None  # No warmup - constant LR like author

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        logger.info(f"Trainer initialized on {device}")
        logger.info(
            f"Learning rate: {learning_rate}, Warmup steps: {warmup_steps}")
        logger.info(f"Accumulation steps: {accumulation_steps}")
        logger.info(f"Checkpoint dir: {checkpoint_dir}")

    def compute_pairwise_loss(
        self,
        query_embeddings: torch.Tensor,
        pos_doc_embeddings: torch.Tensor,
        neg_doc_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise softmax cross-entropy loss.

        For each query:
        - Score against positive document: s_pos = MaxSim(q, d+)
        - Score against negative documents: s_neg = MaxSim(q, d-)
        - Loss = -log(exp(s_pos) / (exp(s_pos) + sum(exp(s_neg))))

        Args:
            query_embeddings: [batch_size, Nq, dim]
            pos_doc_embeddings: [batch_size, Nd_pos, dim]
            neg_doc_embeddings: [batch_size * num_negs, Nd_neg, dim]

        Returns:
            loss: Scalar loss value
        """
        # Get the actual model (handle DataParallel wrapper)
        model = self.model.module if self.multi_gpu else self.model

        batch_size = query_embeddings.size(0)

        # Compute positive scores: [batch_size]
        pos_scores = model.score(query_embeddings, pos_doc_embeddings)

        # Reshape negatives for scoring
        # neg_doc_embeddings: [batch_size * num_negs, Nd_neg, dim]
        num_negs = neg_doc_embeddings.size(0) // batch_size

        # Repeat queries for all negatives: [batch_size * num_negs, Nq, dim]
        query_repeated = query_embeddings.repeat_interleave(num_negs, dim=0)

        # Compute negative scores: [batch_size * num_negs]
        neg_scores = model.score(query_repeated, neg_doc_embeddings)

        # Reshape to [batch_size, num_negs]
        neg_scores = neg_scores.view(batch_size, num_negs)

        # Concatenate positive and negative scores: [batch_size, 1 + num_negs]
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)

        # Compute softmax cross-entropy loss
        # Target is always 0 (first position = positive)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(all_scores, targets)

        return loss

    def train_step(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int
    ) -> Dict[str, float]:
        """
        Perform one training step with author's concatenated batch format.

        Args:
            queries: List of query strings (duplicated: [q1, q2, ..., q1, q2, ...])
            documents: List of documents (concatenated: [pos1, pos2, ..., neg1, neg2, ...])
            batch_size: Original batch size (before duplication)

        Returns:
            Dictionary with training metrics
        """
        # Get the actual model (handle DataParallel wrapper)
        model = self.model.module if self.multi_gpu else self.model

        # Encode queries and documents
        query_embeddings = model.query(queries)
        doc_embeddings = model.doc(documents)

        # Compute scores for all query-document pairs
        scores = model.score(query_embeddings, doc_embeddings)
        
        # Reshape to [batch_size, 2] where column 0 = positive, column 1 = negative
        # Author's approach: view(2, -1).permute(1, 0)
        scores = scores.view(2, batch_size).permute(1, 0)

        # Compute loss: target is always 0 (positive document)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(scores, targets)

        # Backward pass (with gradient accumulation)
        loss = loss / self.accumulation_steps
        loss.backward()

        # Compute metrics
        with torch.no_grad():
            pos_scores = scores[:, 0]
            neg_scores = scores[:, 1]

            metrics = {
                'loss': loss.item() * self.accumulation_steps,
                'pos_score': pos_scores.mean().item(),
                'neg_score': neg_scores.mean().item(),
                'score_gap': pos_scores.mean().item() - neg_scores.mean().item()
            }

        return metrics

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'pos_score': 0.0,
                         'neg_score': 0.0, 'score_gap': 0.0}
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch + 1}",
            disable=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Unpack batch (new format with concatenated docs)
            queries = batch['query']
            documents = batch['documents']
            original_batch_size = batch['batch_size']

            # Training step
            metrics = self.train_step(queries, documents, original_batch_size)

            # Accumulate gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update learning rate (during warmup)
                if self.scheduler and self.global_step < self.warmup_steps:
                    self.scheduler.step()

                self.global_step += 1

            # Update epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'gap': f"{metrics['score_gap']:.3f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Logging
            if self.global_step % self.log_every == 0:
                logger.info(
                    f"Step {self.global_step} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Pos: {metrics['pos_score']:.3f} | "
                    f"Neg: {metrics['neg_score']:.3f} | "
                    f"Gap: {metrics['score_gap']:.3f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Validation
            if self.val_dataloader and self.global_step % self.eval_every == 0:
                val_metrics = self.validate()
                logger.info(f"Validation | Loss: {val_metrics['loss']:.4f}")
                self.model.train()

            # Checkpointing
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches

        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate on validation set.

        Returns:
            Dictionary with validation metrics
        """
        if not self.val_dataloader:
            return {}

        self.model.eval()
        val_metrics = {'loss': 0.0, 'pos_score': 0.0,
                       'neg_score': 0.0, 'score_gap': 0.0}
        num_batches = 0

        # Get the actual model (handle DataParallel wrapper)
        model = self.model.module if self.multi_gpu else self.model

        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation", leave=False):
                queries = batch['query']
                documents = batch['documents']
                batch_size = batch['batch_size']

                # Encode
                query_embeddings = model.query(queries)
                doc_embeddings = model.doc(documents)

                # Compute scores
                scores = model.score(query_embeddings, doc_embeddings)
                scores = scores.view(2, batch_size).permute(1, 0)

                # Compute loss
                targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                loss = F.cross_entropy(scores, targets)

                # Compute metrics
                pos_scores = scores[:, 0]
                neg_scores = scores[:, 1]

                val_metrics['loss'] += loss.item()
                val_metrics['pos_score'] += pos_scores.mean().item()
                val_metrics['neg_score'] += neg_scores.mean().item()
                val_metrics['score_gap'] += (pos_scores.mean() - neg_scores.mean()).item()
                num_batches += 1

        # Average metrics
        for key in val_metrics:
            val_metrics[key] /= num_batches

        return val_metrics

    def train(self, num_epochs: int = None, maxsteps: int = None):
        """
        Train for multiple epochs or until maxsteps.

        Args:
            num_epochs: Total number of epochs to train (optional)
            maxsteps: Maximum number of training steps (optional, author's default: 400000)
        """
        if maxsteps is None and num_epochs is None:
            raise ValueError("Must specify either num_epochs or maxsteps")
        
        start_epoch = self.epoch
        start_time = time.time()
        
        if maxsteps:
            logger.info(f"Training for {maxsteps} steps (author's approach)")
            logger.info(f"Starting from step {self.global_step}")
            
            # Train until maxsteps
            while self.global_step < maxsteps:
                self.epoch += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"Epoch {self.epoch}")
                logger.info(f"{'='*80}")
                
                # Train epoch (will stop if maxsteps reached)
                epoch_metrics = self.train_epoch_with_maxsteps(maxsteps)
                
                # Log summary
                elapsed = time.time() - start_time
                logger.info(
                    f"\nStep {self.global_step}/{maxsteps} - Epoch {self.epoch} Summary:\n"
                    f"  Loss: {epoch_metrics['loss']:.4f}\n"
                    f"  Pos Score: {epoch_metrics['pos_score']:.3f}\n"
                    f"  Neg Score: {epoch_metrics['neg_score']:.3f}\n"
                    f"  Score Gap: {epoch_metrics['score_gap']:.3f}\n"
                    f"  Time: {elapsed/60:.1f} min\n"
                )
                
                if self.global_step >= maxsteps:
                    break
        else:
            logger.info(
                f"Training from epoch {start_epoch + 1} to {num_epochs}")
            logger.info(
                f"Total training steps: {len(self.train_dataloader) * (num_epochs - start_epoch) // self.accumulation_steps}")
            
            for epoch in range(start_epoch, num_epochs):
                self.epoch = epoch

                # Train epoch
                epoch_metrics = self.train_epoch()

                # Log epoch summary
                elapsed = time.time() - start_time
                logger.info(
                    f"\nEpoch {epoch + 1}/{num_epochs} Summary:\n"
                    f"  Loss: {epoch_metrics['loss']:.4f}\n"
                    f"  Pos Score: {epoch_metrics['pos_score']:.3f}\n"
                    f"  Neg Score: {epoch_metrics['neg_score']:.3f}\n"
                    f"  Score Gap: {epoch_metrics['score_gap']:.3f}\n"
                    f"  Time: {elapsed/60:.1f} min\n"
                )

                # Validate at end of epoch
                if self.val_dataloader:
                    val_metrics = self.validate()
                    logger.info(
                        f"Validation:\n"
                        f"  Loss: {val_metrics['loss']:.4f}\n"
                        f"  Score Gap: {val_metrics['score_gap']:.3f}\n"
                    )

                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint("best_model.pt")
                        logger.info("  ✅ New best model saved!")

                # Save epoch checkpoint
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

                # Clean up old epoch checkpoints (keep only last 2 + best)
                self._cleanup_old_checkpoints(keep_last=2)

        logger.info(
            f"Training completed in {(time.time() - start_time)/60:.1f} minutes")
        
        # Save final checkpoint
        self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
        # Save as best_model.pt (since no validation, this is the final model)
        self.save_checkpoint("best_model.pt")
        logger.info("Final checkpoint saved.")
    
    def train_epoch_with_maxsteps(self, maxsteps: int) -> Dict[str, float]:
        """
        Train for one epoch but stop if maxsteps is reached.

        Args:
            maxsteps: Maximum number of training steps

        Returns:
            Dictionary with epoch metrics
        """
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'pos_score': 0.0,
                         'neg_score': 0.0, 'score_gap': 0.0}
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch} (stopping at step {maxsteps})",
            disable=False
        )

        for batch_idx, batch in enumerate(progress_bar):
            if self.global_step >= maxsteps:
                break
            
            # Unpack batch (new format with concatenated docs)
            queries = batch['query']
            documents = batch['documents']
            original_batch_size = batch['batch_size']

            # Training step
            metrics = self.train_step(queries, documents, original_batch_size)

            # Accumulate gradients
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update learning rate (during warmup)
                if self.scheduler and self.global_step < self.warmup_steps:
                    self.scheduler.step()

                self.global_step += 1

            # Update epoch metrics
            for key in epoch_metrics:
                epoch_metrics[key] += metrics[key]
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'gap': f"{metrics['score_gap']:.3f}",
                'step': f"{self.global_step}/{maxsteps}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Logging
            if self.global_step % self.log_every == 0:
                logger.info(
                    f"Step {self.global_step}: Loss={metrics['loss']:.4f}, "
                    f"Gap={metrics['score_gap']:.3f}"
                )

            # Validation
            if self.val_dataloader and self.global_step % self.eval_every == 0:
                val_metrics = self.validate()
                logger.info(
                    f"Step {self.global_step} Validation: "
                    f"Loss={val_metrics['loss']:.4f}, "
                    f"Gap={val_metrics['score_gap']:.3f}"
                )

                # Save best model
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint("best_model.pt")
                    logger.info("  ✅ New best model!")

                self.model.train()

            # Save periodic checkpoint
            if self.global_step % self.save_every == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                self._cleanup_old_checkpoints(keep_last=2)
            
            # Special: Save author's 200k milestone checkpoint (never delete)
            if self.global_step == 200000:
                self.save_checkpoint("checkpoint_author_200k_milestone.pt")
                logger.info("🎯 Saved author's 200k step milestone checkpoint!")

        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches

        return epoch_metrics

    def _cleanup_old_checkpoints(self, keep_last: int = 2):
        """
        Delete old epoch and step checkpoints to save disk space.
        Keeps only the last N checkpoints and best_model.pt

        Args:
            keep_last: Number of recent epoch and step checkpoints to keep
        """
        import glob

        # Find all epoch checkpoints
        epoch_pattern = str(self.checkpoint_dir / "checkpoint_epoch_*.pt")
        epoch_checkpoints = sorted(glob.glob(epoch_pattern))

        if len(epoch_checkpoints) > keep_last:
            # Delete older epoch checkpoints (keep most recent)
            for old_checkpoint in epoch_checkpoints[:-keep_last]:
                try:
                    Path(old_checkpoint).unlink()
                    logger.info(
                        f"Deleted old checkpoint: {Path(old_checkpoint).name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {old_checkpoint}: {e}")

        # Find all step checkpoints (excluding milestone checkpoints)
        step_pattern = str(self.checkpoint_dir / "checkpoint_step_*.pt")
        step_checkpoints = sorted(glob.glob(step_pattern))
        
        # Filter out milestone checkpoints (never delete these)
        milestone_pattern = str(self.checkpoint_dir / "checkpoint_author_*_milestone.pt")
        milestone_checkpoints = set(glob.glob(milestone_pattern))
        step_checkpoints = [ckpt for ckpt in step_checkpoints if ckpt not in milestone_checkpoints]

        if len(step_checkpoints) > keep_last:
            # Delete older step checkpoints (keep most recent)
            for old_checkpoint in step_checkpoints[:-keep_last]:
                try:
                    Path(old_checkpoint).unlink()
                    logger.info(
                        f"Deleted old checkpoint: {Path(old_checkpoint).name}")
                except Exception as e:
                    logger.warning(f"Failed to delete {old_checkpoint}: {e}")

    def save_checkpoint(self, filename: str):
        """
        Save training checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': {
                'learning_rate': self.learning_rate,
                'warmup_steps': self.warmup_steps,
                'accumulation_steps': self.accumulation_steps,
                'max_grad_norm': self.max_grad_norm
            }
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model_to_load = self.model.module if self.multi_gpu else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        logger.info(
            f"Resuming from step {self.global_step}, epoch {self.epoch}")


if __name__ == "__main__":
    """
    Test training loop with dummy data.
    """
    print("Testing ColBERT training loop...")

    # Create model
    model = ColBERT()
    print(
        f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'query': f"sample query {idx}",
                'pos_doc': f"relevant document for query {idx}",
                'neg_docs': [
                    f"negative document {idx}-1",
                    f"negative document {idx}-2"
                ]
            }

    # Collate function
    def collate_fn(batch):
        return {
            'query': [item['query'] for item in batch],
            'pos_doc': [item['pos_doc'] for item in batch],
            'neg_docs': [item['neg_docs'] for item in batch]
        }

    # Create dataloaders
    train_dataset = DummyDataset(100)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = DummyDataset(20)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        collate_fn=collate_fn
    )

    # Create trainer
    trainer = ColBERTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=3e-6,
        warmup_steps=10,
        accumulation_steps=2,
        log_every=5,
        eval_every=20,
        save_every=20,
        checkpoint_dir="models/checkpoints/test"
    )

    print("\nTraining for 1 epoch (test)...")
    trainer.train(num_epochs=1)

    print("\n✅ Training loop test completed!")
