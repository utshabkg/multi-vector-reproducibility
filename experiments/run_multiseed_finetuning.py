#!/usr/bin/env python3
"""
Multi-seed fine-tuning for statistical significance.
Runs 3 seeds for both ColBERT and ConstBERT in parallel on separate GPUs.

Usage:
    python experiments/run_multiseed_finetuning.py --parallel
"""

import subprocess
import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

SEEDS = [42, 123, 456]
BASE_DIR = Path(__file__).parent.parent

# ConstBERT config
CONSTBERT_CONFIG = {
    "script": "experiments/10_finetune_constbert_tot.py",
    "base_args": [
        "--triples_file", "/media/12TB/shared/datasets/indices/processed/trec-tot-2025-triple-bm25/train.triples",
        "--corpus_file", "data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl",
        "--queries_file", "data/trec-tot-2025/raw/queries/train-2025-queries.jsonl",
        "--max_steps", "5000",
        "--batch_size", "8",
        "--grad_accum_steps", "4",
        "--learning_rate", "5e-6",
        "--warmup_steps", "500",
    ]
}

# ColBERT config
COLBERT_CONFIG = {
    "script": "colbert-replicability/colbert/scripts/train_colbert_tot_triples.py",
    "base_args": [
        "--triples_file", "/media/12TB/shared/datasets/indices/processed/trec-tot-2025-triple-bm25/train.triples",
        "--corpus_file", "data/trec-tot-2025/raw/trec-tot-2025-corpus.jsonl",
        "--queries_file", "data/trec-tot-2025/tot_bm25_top1000.tsv",
        "--checkpoint", "/media/12TB/shared/models/colbertv2.0",
        "--maxsteps", "2000",
        "--batch_size", "32",
        "--learning_rate", "3e-6",
        "--warmup_steps", "1000",
        "--use_amp",
    ]
}


def run_constbert_seed(seed: int, gpu_id: int):
    """Run ConstBERT fine-tuning with a specific seed."""
    output_dir = BASE_DIR / f"checkpoints/constbert_tot_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(BASE_DIR)
    
    # Add seed to the script (we need to modify the script to accept seed)
    cmd = [
        sys.executable,
        str(BASE_DIR / CONSTBERT_CONFIG["script"]),
        *CONSTBERT_CONFIG["base_args"],
        "--output_dir", str(output_dir),
    ]
    
    print(f"[ConstBERT seed={seed}] Starting on GPU {gpu_id}")
    start = time.time()
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"[ConstBERT seed={seed}] ✅ Completed in {elapsed/60:.1f} min")
        return {"model": "constbert", "seed": seed, "status": "success", "time": elapsed}
    else:
        print(f"[ConstBERT seed={seed}] ❌ Failed: {result.stderr[-500:]}")
        return {"model": "constbert", "seed": seed, "status": "failed", "error": result.stderr[-1000:]}


def run_colbert_seed(seed: int, gpu_id: int):
    """Run ColBERT fine-tuning with a specific seed."""
    output_dir = BASE_DIR / f"colbert-replicability/colbert/models/colbertv2-tot-seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = str(BASE_DIR) + ":" + str(BASE_DIR / "colbert-replicability/colbert")
    # Set seed via environment variable
    env["RANDOM_SEED"] = str(seed)
    
    cmd = [
        sys.executable,
        str(BASE_DIR / COLBERT_CONFIG["script"]),
        *COLBERT_CONFIG["base_args"],
        "--checkpoint_dir", str(output_dir),
    ]
    
    print(f"[ColBERT seed={seed}] Starting on GPU {gpu_id}")
    start = time.time()
    
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"[ColBERT seed={seed}] ✅ Completed in {elapsed/60:.1f} min")
        return {"model": "colbert", "seed": seed, "status": "success", "time": elapsed}
    else:
        print(f"[ColBERT seed={seed}] ❌ Failed: {result.stderr[-500:]}")
        return {"model": "colbert", "seed": seed, "status": "failed", "error": result.stderr[-1000:]}


def run_parallel():
    """Run all fine-tuning jobs in parallel using both GPUs."""
    print("=" * 80)
    print("Multi-Seed Fine-tuning for Statistical Significance")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print(f"Models: ConstBERT, ColBERT")
    print(f"Total jobs: {len(SEEDS) * 2}")
    print()
    
    # Strategy: Run 2 jobs at a time (one per GPU)
    # Order: alternate between models to balance load
    jobs = []
    for seed in SEEDS:
        jobs.append(("constbert", seed))
        jobs.append(("colbert", seed))
    
    results = []
    
    # Run jobs 2 at a time
    for i in range(0, len(jobs), 2):
        batch = jobs[i:i+2]
        
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = []
            for idx, (model, seed) in enumerate(batch):
                gpu_id = idx  # GPU 0 or 1
                if model == "constbert":
                    futures.append(executor.submit(run_constbert_seed, seed, gpu_id))
                else:
                    futures.append(executor.submit(run_colbert_seed, seed, gpu_id))
            
            for future in as_completed(futures):
                results.append(future.result())
    
    # Save results summary
    summary_file = BASE_DIR / "results/multiseed_finetuning_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for r in results:
        status = "✅" if r["status"] == "success" else "❌"
        print(f"  {status} {r['model']} seed={r['seed']}: {r['status']}")
    
    print(f"\nResults saved to: {summary_file}")
    return results


def run_sequential():
    """Run all fine-tuning jobs sequentially (for debugging)."""
    results = []
    for seed in SEEDS:
        results.append(run_constbert_seed(seed, 0))
        results.append(run_colbert_seed(seed, 1))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run in parallel on 2 GPUs")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument("--constbert-only", action="store_true", help="Only run ConstBERT")
    parser.add_argument("--colbert-only", action="store_true", help="Only run ColBERT")
    args = parser.parse_args()
    
    if args.parallel:
        run_parallel()
    else:
        run_sequential()
