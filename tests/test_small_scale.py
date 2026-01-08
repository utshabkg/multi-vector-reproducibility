"""
Small-scale validation test: run retrieval on 100 queries with 10K passages.
Verifies the complete pipeline before running full-scale experiment.
"""
import sys
from pathlib import Path
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loaders import MSMARCODataLoader
from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import evaluate_retrieval, print_evaluation_results

print("="*80)
print("Small-Scale Validation Test")
print("Testing with 10K documents and 100 queries")
print("="*80)

# 1. Load small subset of data
print("\n[1/5] Loading small data subset...")
data_loader = MSMARCODataLoader()

# Load all queries and qrels (small enough)
all_queries = data_loader.load_queries("dev")
all_qrels = data_loader.load_qrels("dev")

# Select first 100 queries
query_ids = list(all_queries.keys())[:100]
queries = {qid: all_queries[qid] for qid in query_ids}
qrels = {qid: all_qrels[qid] for qid in query_ids if qid in all_qrels}

print(f"Selected {len(queries)} queries, {len(qrels)} with qrels")

# Load first 10K passages
collection = {}
with open("/media/12TB/shared/datasets/raw/msmarco-passage/collection.tsv", 'r') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        pid, passage = line.rstrip('\n').split('\t')
        collection[pid] = passage

print(f"Loaded {len(collection)} passages")

# 2. Initialize model
print("\n[2/5] Initializing ConstBERT model...")
device = "cuda"  # Use GPU for faster encoding
model = ConstBERTWrapper(
    model_name="pinecone/ConstBERT",
    device=device,
    batch_size=32
)

# 3. Encode documents
print("\n[3/5] Encoding documents...")
doc_ids = list(collection.keys())
doc_texts = [collection[doc_id] for doc_id in doc_ids]

start_time = time.time()
doc_embeddings = model.encode_documents(doc_texts, show_progress=True)
encode_time = time.time() - start_time

print(f"Encoding completed in {encode_time:.2f} seconds")
print(f"Document embeddings shape: {doc_embeddings.shape}")
print(f"Expected: ({len(doc_ids)}, 32, 128)")

# Verify shape
assert doc_embeddings.shape == (len(doc_ids), 32, 128), "Unexpected embedding shape!"

# 4. Build index and retrieve
print("\n[4/5] Building FAISS index and running retrieval...")
index = ConstBERTFAISSIndex(use_gpu=True)
index.add_documents(doc_ids, doc_embeddings)

print(f"Index size: {index.get_index_size_mb():.2f} MB")

# Encode queries
query_texts = [queries[qid] for qid in query_ids]
query_embeddings = model.encode_queries(query_texts, show_progress=True)
print(f"Query embeddings shape: {query_embeddings.shape}")

# Retrieve top-k documents for each query
k = 100
results = {}
retrieval_start = time.time()

print(f"\nRetrieving with FAISS (candidate_mult=3, VECTORIZED)...")
for i, qid in enumerate(query_ids):
    query_emb = query_embeddings[i]  # Single query (num_tokens, dim)
    top_doc_ids, scores = index.search(query_emb, k=k, candidate_mult=3)
    # Convert to format expected by evaluation: list of (doc_id, score) tuples
    results[qid] = [(doc_id, float(score)) for doc_id, score in zip(top_doc_ids[0], scores[0])]

retrieval_time = time.time() - retrieval_start
avg_latency = (retrieval_time / len(query_ids)) * 1000  # ms per query

print(f"Retrieval completed in {retrieval_time:.2f} seconds")
print(f"Average latency: {avg_latency:.2f} ms/query")

# 5. Evaluate
print("\n[5/5] Evaluating results...")
metrics = evaluate_retrieval(results, qrels)
print_evaluation_results(metrics)

print("\n" + "="*80)
print("✓ Small-scale validation PASSED")
print("="*80)
print("\nObservations:")
print(f"- Encoding speed: {len(doc_ids)/encode_time:.1f} docs/sec")
print(f"- Retrieval speed: {avg_latency:.2f} ms/query")
print(f"- Model uses C=32 vectors per document (as expected)")
print(f"- Embedding dimension: 128 (as expected)")
print("\nNote: Metrics on 10K docs are NOT comparable to paper (uses 8.8M docs)")
print("This test only validates that the pipeline works correctly.")
print("\nReady to run full experiment: python experiments/exp1_dev_eval.py")
