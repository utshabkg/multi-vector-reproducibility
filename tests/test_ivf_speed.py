"""Test IVF index speed vs Flat index."""
import sys
sys.path.append('/home/ugdf8/IRIS/dev/reproduce/constbert-reproduce')

import numpy as np
import faiss
import time

# Simulate ConstBERT embeddings
num_docs = 100000  # Test with 100K docs (vs 8.8M in full)
C = 32
dim = 128

print("=" * 80)
print(f"FAISS IVF Speed Test")
print(f"Documents: {num_docs:,}, Vectors: {num_docs * C:,}, Dim: {dim}")
print("=" * 80)

# Generate random embeddings
print("\nGenerating random embeddings...")
doc_embs = np.random.randn(num_docs, C, dim).astype('float32')
flat_embs = doc_embs.reshape(num_docs * C, dim)

# Generate query
query_embs = np.random.randn(20, dim).astype('float32')  # 20 tokens

# Test 1: Flat index
print("\n[Test 1] IndexFlatIP (exact search):")
index_flat = faiss.IndexFlatIP(dim)
index_flat.add(flat_embs)

start = time.time()
for _ in range(10):
    _, indices = index_flat.search(query_embs, 100)  # 100 vectors per token
elapsed_flat = (time.time() - start) / 10 * 1000
print(f"  Time per query: {elapsed_flat:.1f} ms")

# Test 2: IVF index
print("\n[Test 2] IndexIVFFlat (approximate search):")
nlist = num_docs // 100  # ~100 vectors per cluster
quantizer = faiss.IndexFlatIP(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

print(f"  Training with {nlist} clusters...")
index_ivf.train(flat_embs)
index_ivf.add(flat_embs)

# Test different nprobe values
for nprobe in [1, 4, 16, 64, 256]:
    if nprobe > nlist:
        continue
    index_ivf.nprobe = nprobe
    
    start = time.time()
    for _ in range(10):
        _, indices = index_ivf.search(query_embs, 100)
    elapsed_ivf = (time.time() - start) / 10 * 1000
    
    speedup = elapsed_flat / elapsed_ivf
    print(f"  nprobe={nprobe:<4} -> {elapsed_ivf:6.1f} ms/query ({speedup:.1f}x faster)")

print("\n" + "=" * 80)
print("Scaling estimate for 8.8M documents:")
scale_factor = 8841823 / 100000
print(f"  If linear scaling: {elapsed_flat * scale_factor / 1000:.1f} seconds per query")
print(f"  With IVF (nprobe=256): estimate ~{elapsed_ivf * np.log(scale_factor) / 1000:.1f}s per query")
print("=" * 80)
