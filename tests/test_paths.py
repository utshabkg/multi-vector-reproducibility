"""
Quick test to verify path configurations before full run.
"""
import os
from pathlib import Path

print("Testing path configurations...")
print("="*60)

# Test 1: Check environment variables
print("\n[1] Environment Variables:")
print(f"  HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")

# Test 2: Check directories exist/writable
print("\n[2] Directory Checks:")

dirs_to_check = {
    "Models": "/media/12TB/shared/models/huggingface",
    "Embeddings": "/media/12TB/shared/datasets/processed/msmarco-passage",
    "Index": "/media/12TB/shared/datasets/indices",
    "Results": "./results"
}

for name, path in dirs_to_check.items():
    p = Path(path)
    exists = p.exists()
    
    if not exists:
        print(f"  {name}: Creating {path}...")
        p.mkdir(parents=True, exist_ok=True)
        exists = p.exists()
    
    writable = os.access(p, os.W_OK) if exists else False
    status = "✓" if (exists and writable) else "✗"
    print(f"  {status} {name}: {path} (exists={exists}, writable={writable})")

print("\n[3] Testing argument defaults:")
print(f"  Data dir: /media/12TB/shared/datasets/raw/msmarco-passage")
print(f"  Embeddings dir: /media/12TB/shared/datasets/processed/msmarco-passage")
print(f"  Index dir: /media/12TB/shared/datasets/indices")
print(f"  Results dir: ./results")

print("\n" + "="*60)
print("✓ Path configuration test complete!")
print("\nReady to run: python experiments/exp1_dev_eval.py")
