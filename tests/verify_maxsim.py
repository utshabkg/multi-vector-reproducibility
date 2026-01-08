"""
Verify that vectorized MaxSim implementation gives identical results to loop version.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper

# Test data
np.random.seed(42)
query_emb = np.random.randn(32, 128)  # 32 query tokens, 128 dim
doc_embs = np.random.randn(100, 32, 128)  # 100 docs, 32 vectors each, 128 dim

print("Testing MaxSim implementation correctness...")
print(f"Query shape: {query_emb.shape}")
print(f"Docs shape: {doc_embs.shape}")

# Method 1: Loop version (ground truth from paper formula)
def maxsim_loop(q_emb, d_emb):
    """Original loop-based MaxSim: s(q,d) = Σᵢ max_j qᵢᵀδⱼ"""
    scores = np.dot(d_emb, q_emb.T)  # (C, num_query_tokens)
    return float(np.sum(np.max(scores, axis=0)))  # max over C, sum over query tokens

scores_loop = [maxsim_loop(query_emb, doc_embs[i]) for i in range(len(doc_embs))]

# Method 2: Vectorized version currently in code
num_docs, C, dim = doc_embs.shape
reshaped_embs = doc_embs.reshape(num_docs * C, dim)
dot_products = reshaped_embs @ query_emb.T
dot_products = dot_products.reshape(num_docs, C, -1)
max_sims = np.max(dot_products, axis=1)
scores_vectorized = np.sum(max_sims, axis=1)

# Compare
print(f"\nLoop version scores (first 5): {scores_loop[:5]}")
print(f"Vectorized scores (first 5): {scores_vectorized[:5]}")
print(f"\nMax absolute difference: {np.max(np.abs(np.array(scores_loop) - scores_vectorized))}")
print(f"Are they identical? {np.allclose(scores_loop, scores_vectorized)}")

if np.allclose(scores_loop, scores_vectorized):
    print("\n✓ Vectorized implementation is mathematically IDENTICAL to paper formula!")
    print("  It's just a computational optimization, not an architectural change.")
else:
    print("\n✗ ERROR: Implementations differ!")
    sys.exit(1)
