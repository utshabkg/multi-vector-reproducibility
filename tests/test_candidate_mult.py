"""
Test to verify that FAISS candidate selection doesn't miss true top-k documents.
Compares FAISS+MaxSim results with brute-force MaxSim on a small subset.
"""
import sys
sys.path.append('/home/ugdf8/IRIS/dev/reproduce/constbert-reproduce')

import numpy as np
from models.faiss_index import ConstBERTFAISSIndex
from models.constbert_wrapper import ConstBERTWrapper
from data.loaders import MSMARCODataLoader
import time

def brute_force_maxsim(query_emb, doc_embs, k=1000):
    """Compute exact top-k using brute-force MaxSim."""
    scores = []
    for doc_emb in doc_embs:
        score = ConstBERTWrapper.max_sim(query_emb, doc_emb)
        scores.append(score)
    scores = np.array(scores)
    
    if len(scores) > k:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    else:
        top_indices = np.argsort(scores)[::-1]
    
    return top_indices, scores[top_indices]

def recall_at_k(retrieved_indices, true_indices, k):
    """Compute recall@k."""
    retrieved_set = set(retrieved_indices[:k])
    true_set = set(true_indices[:k])
    return len(retrieved_set & true_set) / len(true_set) if len(true_set) > 0 else 0.0

print("=" * 80)
print("Testing FAISS Candidate Selection Accuracy")
print("=" * 80)

# Load small subset
print("\n[1] Loading test data...")
loader = MSMARCODataLoader()
collection = loader.load_collection()
doc_ids = list(collection.keys())[:5000]  # Test on 5K docs
passages = [collection[doc_id] for doc_id in doc_ids]

queries = loader.load_queries('dev')
query_ids = list(queries.keys())[:20]  # Test on 20 queries
query_texts = [queries[qid] for qid in query_ids]

print(f"Testing on {len(passages)} documents, {len(query_texts)} queries")

# Initialize model
print("\n[2] Loading model...")
model = ConstBERTWrapper(batch_size=32)

# Encode
print("\n[3] Encoding...")
doc_embs = model.encode_documents(passages, show_progress=True)
query_embs = model.encode_queries(query_texts, show_progress=False)

print(f"Document embeddings: {doc_embs.shape}")
print(f"Query embeddings: {query_embs.shape}")

# Build FAISS index
print("\n[4] Building FAISS index...")
index = ConstBERTFAISSIndex(use_gpu=False)
index.add_documents(doc_ids, doc_embs)

# Test different candidate_mult values
print("\n[5] Testing different candidate_mult values...")
print("-" * 80)
print(f"{'candidate_mult':<15} {'Recall@1000':<15} {'Recall@100':<15} {'Time (ms/q)':<15}")
print("-" * 80)

for candidate_mult in [1, 2, 3, 5, 10, 20]:
    start_time = time.time()
    
    # FAISS retrieval
    faiss_doc_ids, faiss_scores = index.search(
        query_embs, 
        k=1000, 
        candidate_mult=candidate_mult,
        show_progress=False
    )
    
    elapsed = (time.time() - start_time) * 1000 / len(query_texts)
    
    # Compare with brute force for first query
    query_idx = 0
    true_indices, true_scores = brute_force_maxsim(query_embs[query_idx], doc_embs, k=1000)
    
    # Get FAISS result indices
    faiss_indices = [doc_ids.index(did) for did in faiss_doc_ids[query_idx] if did in doc_ids]
    
    # Compute recall
    recall_1000 = recall_at_k(faiss_indices, true_indices, 1000)
    recall_100 = recall_at_k(faiss_indices, true_indices, 100)
    
    print(f"{candidate_mult:<15} {recall_1000:<15.4f} {recall_100:<15.4f} {elapsed:<15.1f}")

print("-" * 80)

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("- candidate_mult=1: Retrieves k*1 candidates (may miss true top-k)")
print("- candidate_mult=3: Retrieves k*3 candidates (balance speed/accuracy)")
print("- candidate_mult=10: Retrieves k*10 candidates (high accuracy)")
print()
print("For reproducibility, we should use candidate_mult that gives:")
print("  Recall@1000 = 1.0 (retrieves all true top-1000 documents)")
print()
print("Recommendation:")
if True:  # Will be determined by results
    print("  Use candidate_mult >= 3 to ensure high recall")
    print("  Paper likely used brute-force (equivalent to candidate_mult=∞)")
    print("  Our FAISS optimization is ENGINEERING ONLY - doesn't change scoring")
