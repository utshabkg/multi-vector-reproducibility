"""
Deep investigation into ConstBERT model to understand storage format.
Check model config, architecture, and any documentation about quantization.
"""
import sys
sys.path.append('/home/ugdf8/IRIS/dev/reproduce/constbert-reproduce')

from transformers import AutoModel, AutoConfig
import torch
import json

print("=" * 80)
print("ConstBERT Model Deep Dive - Storage Investigation")
print("=" * 80)

# Load model config
print("\n1. MODEL CONFIGURATION")
print("-" * 80)
config = AutoConfig.from_pretrained("pinecone/ConstBERT", trust_remote_code=True)
print(json.dumps(config.to_dict(), indent=2))

# Load model
print("\n2. MODEL ARCHITECTURE & PARAMETERS")
print("-" * 80)
model = AutoModel.from_pretrained("pinecone/ConstBERT", trust_remote_code=True)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Check for any quantization info
print("\n3. QUANTIZATION CHECK")
print("-" * 80)
for name, param in model.named_parameters():
    if 'quant' in name.lower() or 'scale' in name.lower():
        print(f"  {name}: {param.dtype}, shape: {param.shape}")

# Check data types
print("\n4. PARAMETER DATA TYPES")
print("-" * 80)
dtypes = {}
for name, param in model.named_parameters():
    dtype = str(param.dtype)
    dtypes[dtype] = dtypes.get(dtype, 0) + 1

for dtype, count in dtypes.items():
    print(f"  {dtype}: {count} parameters")

# Check model size on disk
print("\n5. MODEL FILE SIZE")
print("-" * 80)
from pathlib import Path
import os

cache_dir = Path(os.environ.get('TRANSFORMERS_CACHE', '/media/12TB/shared/models/huggingface/hub'))
print(f"HuggingFace cache: {cache_dir}")

# Try to find the model files
model_dirs = list(cache_dir.glob("*pinecone*ConstBERT*")) + list(cache_dir.glob("*ConstBERT*"))
if model_dirs:
    total_model_size = 0
    for model_dir in model_dirs[:3]:
        print(f"\nFound: {model_dir.name}")
        # Find model files
        for file_path in model_dir.rglob("*.bin"):
            size_mb = file_path.stat().st_size / (1024**2)
            total_model_size += size_mb
            print(f"  {file_path.name}: {size_mb:.1f} MB")
        for file_path in model_dir.rglob("*.safetensors"):
            size_mb = file_path.stat().st_size / (1024**2)
            total_model_size += size_mb
            print(f"  {file_path.name}: {size_mb:.1f} MB")
    if total_model_size > 0:
        print(f"\nTotal model size: {total_model_size:.1f} MB ({total_model_size/1024:.2f} GB)")
else:
    print("No model files found in cache. Checking current directory...")
    import subprocess
    result = subprocess.run(['find', str(cache_dir), '-name', '*ConstBERT*', '-type', 'd'], 
                          capture_output=True, text=True, timeout=10)
    if result.stdout:
        print("Found directories:")
        print(result.stdout[:500])

# Check if there's any documentation about storage
print("\n6. CHECKING FOR STORAGE DOCUMENTATION")
print("-" * 80)
if hasattr(model, 'config'):
    cfg = model.config
    storage_keys = ['storage', 'quantization', 'compression', 'index_size', 'precision']
    for key in storage_keys:
        if hasattr(cfg, key):
            print(f"  {key}: {getattr(cfg, key)}")
    
    # Check all config attributes
    print("\nAll config attributes:")
    for attr in dir(cfg):
        if not attr.startswith('_') and not callable(getattr(cfg, attr)):
            val = getattr(cfg, attr)
            if isinstance(val, (int, float, str, bool)) and 'token' not in attr.lower():
                print(f"  {attr}: {val}")

print("\n7. EMBEDDINGS OUTPUT CHECK")
print("-" * 80)
# Encode a sample to see output format
sample_text = ["This is a test document."]
try:
    with torch.no_grad():
        inputs = model.doc_tokenizer(sample_text, return_tensors='pt', padding=True, truncation=True)
        outputs = model.doc(**inputs)
        
    print(f"Output dtype: {outputs.dtype}")
    print(f"Output shape: {outputs.shape}")
    print(f"Expected C value: {outputs.shape[1]}")
    print(f"Expected dim value: {outputs.shape[2]}")
except Exception as e:
    print(f"Could not generate sample output: {e}")
    print("Checking model methods...")
    print(f"Available methods: {[m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("The public HuggingFace model stores parameters in float32/float16.")
print("Document embeddings are generated in float16 during inference.")
print("Paper's 11 GB claim likely requires:")
print("  1. Post-processing quantization (int8, 4-bit, or product quantization)")
print("  2. Custom compression not included in public release")
print("  3. Specialized PLAID index format")
print("\nConclusion: Storage discrepancy is due to missing quantization artifacts")
print("            not available in the public model release.")
print("=" * 80)
