"""
Test script to verify ConstBERT setup and basic functionality.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("ConstBERT Setup Verification Test")
print("="*80)

# Test 1: Check imports
print("\n[Test 1] Checking package imports...")
try:
    import torch
    import transformers
    import numpy as np
    import pandas as pd
    from data.loaders import MSMARCODataLoader
    from models.constbert_wrapper import ConstBERTWrapper
    from evaluation.metrics import evaluate_retrieval
    print("✓ All imports successful")
    print(f"  - PyTorch: {torch.__version__}")
    print(f"  - Transformers: {transformers.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check data availability
print("\n[Test 2] Checking data availability...")
try:
    data_loader = MSMARCODataLoader()
    # Just check if files exist, don't load all
    collection_path = Path("/media/12TB/shared/datasets/raw/msmarco-passage/collection.tsv")
    queries_path = Path("/media/12TB/shared/datasets/raw/msmarco-passage/queries.dev.small.tsv")
    qrels_path = Path("/media/12TB/shared/datasets/raw/msmarco-passage/qrels.dev.small.tsv")
    
    assert collection_path.exists(), "collection.tsv not found"
    assert queries_path.exists(), "queries.dev.small.tsv not found"
    assert qrels_path.exists(), "qrels.dev.small.tsv not found"
    
    print("✓ All data files accessible")
    print(f"  - Collection: {collection_path}")
    print(f"  - Queries: {queries_path}")
    print(f"  - Qrels: {qrels_path}")
except Exception as e:
    print(f"✗ Data check failed: {e}")
    sys.exit(1)

# Test 3: Load small data sample
print("\n[Test 3] Loading small data sample...")
try:
    queries = data_loader.load_queries("dev")
    qrels = data_loader.load_qrels("dev")
    print(f"✓ Loaded {len(queries)} dev queries, {len(qrels)} qrels")
except Exception as e:
    print(f"✗ Data loading failed: {e}")
    sys.exit(1)

# Test 4: Try loading ConstBERT model
print("\n[Test 4] Testing ConstBERT model loading...")
print("This may take a minute to download the model from HuggingFace...")
try:
    model = ConstBERTWrapper(
        model_name="pinecone/ConstBERT",
        device="cpu",  # Use CPU for quick test
        batch_size=2
    )
    print("✓ Model loaded successfully")
    print(f"  - Device: {model.device}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    print("\nNote: If model download fails, you may need to:")
    print("  1. Check your internet connection")
    print("  2. Login to HuggingFace: huggingface-cli login")
    sys.exit(1)

# Test 5: Test encoding on small samples
print("\n[Test 5] Testing encoding functionality...")
try:
    test_queries = ["what is machine learning", "how does neural network work"]
    test_docs = ["Machine learning is a subset of artificial intelligence",
                 "Neural networks are computing systems inspired by biological neural networks"]
    
    print("Encoding test queries...")
    query_embeddings = model.encode_queries(test_queries, show_progress=False)
    print(f"✓ Query encoding works. Shape: {query_embeddings.shape}")
    
    print("Encoding test documents...")
    doc_embeddings = model.encode_documents(test_docs, show_progress=False)
    print(f"✓ Document encoding works. Shape: {doc_embeddings.shape}")
    print(f"  - Fixed vectors per doc (C): {doc_embeddings.shape[1]}")
    print(f"  - Embedding dimension: {doc_embeddings.shape[2]}")
except Exception as e:
    print(f"✗ Encoding failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✓ ALL TESTS PASSED - Setup is working correctly!")
print("="*80)
print("\nNext steps:")
print("1. Run small-scale validation: python tests/test_small_scale.py")
print("2. Run full experiment: python experiments/exp1_dev_eval.py")
