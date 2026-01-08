"""
Analyze storage requirements and compare with paper claims.
Calculate theoretical vs actual storage for ConstBERT32.
"""
import sys
sys.path.append('/home/ugdf8/IRIS/dev/reproduce/constbert-reproduce')

import numpy as np
from pathlib import Path
import pickle

print("=" * 80)
print("ConstBERT Storage Analysis")
print("=" * 80)

# Paper claims
PAPER_CONSTBERT32_SIZE_GB = 11
PAPER_COLBERT_SIZE_GB = 22
NUM_DOCS = 8841823
C = 32
DIM = 128

print("\n1. THEORETICAL STORAGE CALCULATION")
print("-" * 80)

# Float32 (4 bytes per value)
bytes_per_doc_float32 = C * DIM * 4
total_bytes_float32 = NUM_DOCS * bytes_per_doc_float32
total_gb_float32 = total_bytes_float32 / (1024**3)

print(f"Number of documents: {NUM_DOCS:,}")
print(f"Vectors per document (C): {C}")
print(f"Embedding dimension: {DIM}")
print(f"\nWith float32 (4 bytes):")
print(f"  Bytes per document: {bytes_per_doc_float32:,}")
print(f"  Total storage: {total_gb_float32:.2f} GB")

# Float16 (2 bytes per value)
bytes_per_doc_float16 = C * DIM * 2
total_bytes_float16 = NUM_DOCS * bytes_per_doc_float16
total_gb_float16 = total_bytes_float16 / (1024**3)

print(f"\nWith float16 (2 bytes):")
print(f"  Bytes per document: {bytes_per_doc_float16:,}")
print(f"  Total storage: {total_gb_float16:.2f} GB")

# Int8 (1 byte per value)
bytes_per_doc_int8 = C * DIM * 1
total_bytes_int8 = NUM_DOCS * bytes_per_doc_int8
total_gb_int8 = total_bytes_int8 / (1024**3)

print(f"\nWith int8 quantization (1 byte):")
print(f"  Bytes per document: {bytes_per_doc_int8:,}")
print(f"  Total storage: {total_gb_int8:.2f} GB")

print("\n2. ACTUAL STORAGE ON DISK")
print("-" * 80)

# Check our files
embeddings_path = Path("/media/12TB/shared/datasets/processed/msmarco-passage/constbert_msmarco_embeddings.npy")
index_faiss_path = Path("/media/12TB/shared/datasets/indices/constbert_msmarco_faiss_index.faiss")
index_pkl_path = Path("/media/12TB/shared/datasets/indices/constbert_msmarco_faiss_index.pkl")

if embeddings_path.exists():
    emb_size_gb = embeddings_path.stat().st_size / (1024**3)
    print(f"Embeddings (.npy): {emb_size_gb:.2f} GB")
    
    # Load and check dtype
    emb = np.load(embeddings_path, mmap_mode='r')
    print(f"  Shape: {emb.shape}")
    print(f"  Dtype: {emb.dtype}")
    print(f"  Expected size: {emb.nbytes / (1024**3):.2f} GB")

if index_faiss_path.exists():
    faiss_size_gb = index_faiss_path.stat().st_size / (1024**3)
    print(f"\nFAISS index (.faiss): {faiss_size_gb:.2f} GB")

if index_pkl_path.exists():
    pkl_size_gb = index_pkl_path.stat().st_size / (1024**3)
    print(f"Metadata (.pkl): {pkl_size_gb:.2f} GB")

total_our_storage = emb_size_gb + faiss_size_gb + pkl_size_gb
print(f"\nTotal our storage: {total_our_storage:.2f} GB")

print("\n3. COMPARISON WITH PAPER")
print("-" * 80)
print(f"Paper claims (ConstBERT32): {PAPER_CONSTBERT32_SIZE_GB} GB")
print(f"Our storage: {total_our_storage:.2f} GB")
print(f"Ratio: {total_our_storage / PAPER_CONSTBERT32_SIZE_GB:.1f}x larger")

print("\n4. HYPOTHESIS FOR DISCREPANCY")
print("-" * 80)
print("Possible reasons for gap:")
print(f"1. Quantization: Paper may use int8 ({total_gb_int8:.1f} GB) or float16 ({total_gb_float16:.1f} GB)")
print(f"2. Compression: Paper may use additional compression (e.g., product quantization)")
print(f"3. Different format: Paper may use optimized binary format vs our FAISS+pickle")
print(f"4. Index-only: Paper may report only retrieval index, not source embeddings")
print(f"5. FAISS overhead: Our IVF index includes clustering metadata")

print("\n5. STORAGE BREAKDOWN")
print("-" * 80)
if embeddings_path.exists() and index_pkl_path.exists():
    print("Our approach stores:")
    print(f"  - Raw embeddings: {emb_size_gb:.1f} GB (for reference/reuse)")
    print(f"  - FAISS IVF index: {faiss_size_gb:.1f} GB (retrieval structure)")
    print(f"  - Metadata: {pkl_size_gb:.1f} GB (doc IDs, config)")
    print()
    print("Paper's approach likely stores:")
    print(f"  - Quantized embeddings only: ~{PAPER_CONSTBERT32_SIZE_GB} GB")
    print(f"  - Optimized for production deployment")

print("\n6. FAIR COMPARISON")
print("-" * 80)
print("For apple-to-apples comparison:")
print(f"  Our raw embeddings (float32): {emb_size_gb:.1f} GB")
print(f"  Paper's claimed size: {PAPER_CONSTBERT32_SIZE_GB} GB")
print(f"  Ratio: {emb_size_gb / PAPER_CONSTBERT32_SIZE_GB:.1f}x")
print()
print("If paper uses float16 or int8 quantization:")
print(f"  float16 would be: {total_gb_float16:.1f} GB (we'd be {emb_size_gb / total_gb_float16:.1f}x)")
print(f"  int8 would be: {total_gb_int8:.1f} GB (we'd be {emb_size_gb / total_gb_int8:.1f}x)")

print("\n" + "=" * 80)
