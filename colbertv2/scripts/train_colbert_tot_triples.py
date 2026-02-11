#!/usr/bin/env python3
"""
Train ColBERT on TREC TOT 2025 dataset with PRE-COMPUTED BM25 TRIPLES.
Adapted from train_colbert_msmarco_triples.py to use BM25-based triples for ToT fine-tuning.
"""

import sys
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# Add parent directories to path to access src modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # Go up to colbert-reproduce root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Go up to replicability

# Now import after path is set
from tot_data_loaders_triples import TOTTriplesDataset, TOTTriplesDatasetValidation
from src.colbert_training import ColBERTTrainer
from src.colbert_model import ColBERT


def main(args):
    print("=" * 80)
    print("ColBERT Fine-Tuning - TREC TOT 2025 with BM25 TRIPLES")
    print("=" * 80)
    print(f"\n🎯 Goal: Fine-tune ColBERT-v2 on ToT dataset")
    print(f"📝 Using BM25-based pre-computed triples")
    print(f"🔍 Better than random negatives!")

    # Set random seeds for reproducibility
    import random
    import numpy as np
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    
    print(f"\n🎲 Random seeds set: 12345")

    # Check CUDA
    if torch.cuda.is_available():
        print(f"\n✅ CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("\n⚠️  CUDA not available, training will be VERY slow")

    # Load datasets
    print("\n" + "=" * 80)
    print("Loading TREC TOT Training Data (BM25 TRIPLES)")
    print("=" * 80)

    train_dataset = TOTTriplesDataset(
        triples_file=args.triples_file,
        corpus_file=args.corpus_file,
        queries_file=args.queries_file,
        max_samples=args.max_samples if args.test_mode else None
    )

    # Validation dataset (if dev data exists)
    val_dataset = None
    if args.val_queries_file and args.val_qrels_file:
        try:
            val_dataset = TOTTriplesDatasetValidation(
                queries_file=args.val_queries_file,
                qrels_file=args.val_qrels_file,
                corpus_file=args.corpus_file,
                max_samples=args.val_samples if args.test_mode else 200,
                negatives_per_query=1
            )
        except Exception as e:
            print(f"\n⚠️  Validation data not available: {e}")
            val_dataset = None

    print(f"\n📊 Dataset Statistics:")
    print(f"   Training triples: {len(train_dataset):,}")
    if val_dataset:
        print(f"   Validation pairs: {len(val_dataset):,}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=TOTTriplesDataset.collate_fn,
        pin_memory=True,
        drop_last=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=TOTTriplesDatasetValidation.collate_fn,
            pin_memory=True
        )

    print(f"\n📦 DataLoader Configuration:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Num workers: {args.num_workers}")
    print(f"   Training batches per epoch: {len(train_loader):,}")
    if val_loader:
        print(f"   Validation batches: {len(val_loader):,}")

    # Initialize model
    print("\n" + "=" * 80)
    print("Initializing ColBERT Model")
    print("=" * 80)

    model = ColBERT(
        bert_model='bert-base-uncased',
        embedding_dim=128,
        query_maxlen=32,
        doc_maxlen=180
    )

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            print(f"\n📦 Loading ColBERT-v2.0 checkpoint from: {checkpoint_path}")
            if checkpoint_path.is_dir():
                # Look for PyTorch model file
                pytorch_model = checkpoint_path / "pytorch_model.bin"
                if pytorch_model.exists():
                    print(f"   Found pytorch_model.bin - attempting to load ColBERT-v2.0 weights")
                    try:
                        checkpoint = torch.load(pytorch_model, map_location='cpu')
                        
                        # Try to load the weights with strict=False to handle mismatches
                        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                        
                        if unexpected_keys:
                            print(f"   ⚠️  Ignored {len(unexpected_keys)} unexpected keys (architecture differences)")
                        if missing_keys:
                            print(f"   ⚠️  {len(missing_keys)} keys were missing and randomly initialized")
                        
                        print("✅ Loaded ColBERT-v2.0 model weights (with some architecture differences)")
                        print("   Fine-tuning will adapt these weights to ToT data")
                        
                    except Exception as e:
                        print(f"   ❌ Failed to load checkpoint: {e}")
                        print("   Starting from BERT base instead")
                else:
                    print(f"   ⚠️  pytorch_model.bin not found in {checkpoint_path}")
                    print("   Starting from BERT base")
            else:
                print(f"   Loading from checkpoint file: {checkpoint_path}")
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    print("✅ Loaded checkpoint successfully (with strict=False)")
                except Exception as e:
                    print(f"   ❌ Failed to load checkpoint: {e}")
                    print("   Starting from BERT base")
        else:
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            print("   Starting from BERT base")
    else:
        print(f"\n🆕 Starting from BERT base (no initial checkpoint specified)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Embedding dimension: {model.embedding_dim}")
    print(f"   Query maxlen: {model.query_maxlen} tokens")
    print(f"   Document maxlen: {model.doc_maxlen} tokens")

    # Initialize trainer
    print("\n" + "=" * 80)
    print("Initializing Trainer")
    print("=" * 80)

    trainer = ColBERTTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        eval_every=args.validate_every,
        save_every=args.checkpoint_every,
        accumulation_steps=args.accumulation_steps
    )

    print(f"\n⚙️  Training Configuration:")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Warmup steps: {args.warmup_steps}")
    print(f"   Gradient accumulation: {args.accumulation_steps} steps")
    print(f"   Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"   Checkpoint dir: {args.checkpoint_dir}")
    print(f"   Log every: {args.log_every} steps")
    if val_loader:
        print(f"   Validate every: {args.validate_every} steps")
    print(f"   Checkpoint every: {args.checkpoint_every} steps")

    # Check for existing checkpoints (resumption)
    checkpoint_dir = Path(args.checkpoint_dir)
    if checkpoint_dir.exists() and not args.restart_training:
        checkpoints = list(checkpoint_dir.glob('checkpoint_step_*.pt'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            print(f"\n📂 Found existing checkpoint: {latest_checkpoint}")
            try:
                trainer.load_checkpoint(str(latest_checkpoint))
                print(f"✅ Resumed from step {trainer.global_step}")
            except Exception as e:
                print(f"⚠️  Could not load checkpoint: {e}")
                print("   Starting from scratch")
        else:
            print("\n🆕 No checkpoints found - starting training from scratch")
    else:
        if args.restart_training:
            print("\n🔄 Restart flag set - ignoring existing checkpoints")
        else:
            print("\n🆕 No checkpoints found - starting training from scratch")

    # Train
    print("\n" + "=" * 80)
    print("Starting Training (ToT Dataset with BM25 Triples)")
    print("=" * 80)
    print(f"\n🎯 Target: Fine-tune on ToT with BM25-based negatives")
    print(f"📈 Expected: Better than random negatives")
    print(f"✅ Using pre-computed BM25 triples")

    trainer.train(maxsteps=args.maxsteps)

    print("\n" + "=" * 80)
    print("✅ Training Complete!")
    print("=" * 80)
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"Best model: {args.checkpoint_dir}/best_model.pt")
    print(f"\nNext steps:")
    print(f"1. Re-index ToT corpus with fine-tuned checkpoint")
    print(f"2. Evaluate on ToT dev1 queries")
    print(f"3. Compare with base ColBERT-v2 performance")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune ColBERT on TREC TOT 2025 with BM25 triples")

    # Data paths
    parser.add_argument('--triples_file', type=str,
                        default='data/processed/trec-tot-2025-triple-bm25/train.triples',
                        help='Path to triples file (qid\\tpos_pid\\tneg_pid)')
    parser.add_argument('--corpus_file', type=str,
                        default='data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl',
                        help='Path to corpus JSONL file')
    parser.add_argument('--queries_file', type=str,
                        default='data/trec-tot-2025/tot_bm25_top1000.tsv',
                        help='Path to queries file (TSV or JSONL)')
    parser.add_argument('--val_queries_file', type=str,
                        default=None,
                        help='Path to validation queries file')
    parser.add_argument('--val_qrels_file', type=str,
                        default=None,
                        help='Path to validation qrels file')
    parser.add_argument('--checkpoint', type=str,
                        default='colbert-ir/colbertv2.0',
                        help='Path to initial checkpoint (ColBERT-v2 base)')
    parser.add_argument('--checkpoint_dir', type=str,
                        default='replicability/colbert/models/colbertv2-fine-tot',
                        help='Directory to save checkpoints')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=3e-6,
                        help='Learning rate')
    parser.add_argument('--maxsteps', type=int, default=2000,
                        help='Maximum training steps')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Warmup steps')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

    # Training configuration
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Logging and checkpointing
    parser.add_argument('--log_every', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--validate_every', type=int, default=1000,
                        help='Validate every N steps')
    parser.add_argument('--checkpoint_every', type=int, default=500,
                        help='Save checkpoint every N steps')

    # Testing
    parser.add_argument('--test_mode', action='store_true',
                        help='Run in test mode with limited data')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Max training samples in test mode')
    parser.add_argument('--val_samples', type=int, default=20,
                        help='Max validation samples in test mode')
    parser.add_argument('--restart_training', action='store_true',
                        help='Ignore existing checkpoints and restart training')

    args = parser.parse_args()
    main(args)
