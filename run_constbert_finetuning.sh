#!/bin/bash
# Master script to fine-tune ConstBERT on TREC ToT 2025
# This runs the complete pipeline: triple generation → fine-tuning → evaluation

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                    ConstBERT ToT Fine-tuning Pipeline                      ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Configuration
TEST_MODE=false
if [[ "$1" == "--test" ]] || [[ "$1" == "--dry-run" ]]; then
    TEST_MODE=true
    echo "🧪 TEST MODE: Running with minimal parameters"
fi

# Paths - using standard ML splits
CORPUS_FILE="/media/12TB/shared/datasets/raw/trec-tot-2025/trec-tot-2025-corpus.jsonl"

# Training data
TRAIN_QUERIES="/media/12TB/shared/datasets/raw/trec-tot-2025/queries/train-2025-queries.jsonl"
TRAIN_QRELS="/media/12TB/shared/datasets/raw/trec-tot-2025/qrel/train-2025-qrel.txt"
TRIPLES_FILE="/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-triple-bm25/train.triples"

# Validation data (monitor during training)
DEV1_QUERIES="/media/12TB/shared/datasets/raw/trec-tot-2025/queries/dev1-2025-queries.jsonl"
DEV1_QRELS="/media/12TB/shared/datasets/raw/trec-tot-2025/qrel/dev1-2025-qrel.txt"

# Test data (final evaluation for paper)
TEST_QUERIES="/media/12TB/shared/datasets/raw/trec-tot-2025/queries/test-2025-queries.jsonl"
TEST_QRELS="/media/12TB/shared/datasets/raw/trec-tot-2025/qrel/test-2025-qrel.txt"

OUTPUT_DIR="checkpoints/constbert_tot_finetuned"

if $TEST_MODE; then
    OUTPUT_DIR="checkpoints/constbert_tot_finetuned_test"
fi

# Create directories
mkdir -p data/tot/finetuning
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# Timestamp for logs
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="logs/constbert_finetuning_${TIMESTAMP}.log"

echo "📝 Configuration:"
echo "   Corpus: $CORPUS_FILE"
echo ""
echo "   Training:"
echo "     Queries: $TRAIN_QUERIES (143 queries)"
echo "     Qrels: $TRAIN_QRELS"
echo "     Triples: $TRIPLES_FILE"
echo ""
echo "   Validation (monitoring):"
echo "     DEV1 Queries: $DEV1_QUERIES (142 queries)"
echo "     DEV1 Qrels: $DEV1_QRELS"
echo ""
echo "   Test (final evaluation):"
echo "     TEST Queries: $TEST_QUERIES (622 queries)"
echo "     TEST Qrels: $TEST_QRELS"
echo ""
echo "   Model output: $OUTPUT_DIR"
echo "   Log file: $LOG_FILE"
echo ""

# Function to log and execute
run_step() {
    local step_name="$1"
    shift
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║ STEP: $step_name"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    "$@" 2>&1 | tee -a "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "❌ ERROR: Step '$step_name' failed!"
        exit 1
    fi
    
    echo "✅ Step '$step_name' completed successfully"
}

# Step 1: Verify training triples exist
if [ ! -f "$TRIPLES_FILE" ]; then
    echo "❌ ERROR: Triples file not found: $TRIPLES_FILE"
    echo "   Please generate triples first."
    exit 1
else
    TRIPLE_COUNT=$(wc -l < "$TRIPLES_FILE")
    echo "✅ Using existing triples file: $TRIPLES_FILE ($TRIPLE_COUNT triples)"
fi

# Step 2: Fine-tune ConstBERT on TRAIN data
FINETUNE_ARGS=(
    --triples_file "$TRIPLES_FILE"
    --corpus_file "$CORPUS_FILE"
    --queries_file "$TRAIN_QUERIES"
    --output_dir "$OUTPUT_DIR"
    --batch_size 8
    --grad_accum_steps 4
    --learning_rate 5e-6
    --warmup_steps 500
    --max_steps 5000
)

if $TEST_MODE; then
    FINETUNE_ARGS+=(--test_mode --max_steps 50)
fi

run_step "Fine-tune ConstBERT on TRAIN split" python -u experiments/10_finetune_constbert_tot.py "${FINETUNE_ARGS[@]}"

# Step 3: Create evaluation script (reusable for both DEV and TEST)
EVAL_SCRIPT="experiments/10_eval_finetuned_constbert.py"

if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "Creating evaluation script..."
    cat > "$EVAL_SCRIPT" << 'EVAL_EOF'
#!/usr/bin/env python3
"""Evaluate fine-tuned ConstBERT on ToT dev1"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex


def load_queries(queries_file):
    """Load queries from JSONL"""
    queries = {}
    with open(queries_file) as f:
        for line in f:
            q = json.loads(line)
            qid = str(q.get('query_id') or q.get('id'))
            text = q.get('query') or q.get('text')
            queries[qid] = text
    return queries


def load_qrels(qrels_file):
    """Load qrels"""
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                if rel > 0:
                    if qid not in qrels:
                        qrels[qid] = []
                    qrels[qid].append(docid)
    return qrels


def compute_mrr(rankings, qrels):
    """Compute MRR@10"""
    mrr_sum = 0.0
    evaluated = 0
    
    for qid, ranking in rankings.items():
        if qid not in qrels:
            continue
        
        relevant = set(qrels[qid])
        for rank, docid in enumerate(ranking[:10], 1):
            if docid in relevant:
                mrr_sum += 1.0 / rank
                break
        
        evaluated += 1
    
    return mrr_sum / evaluated if evaluated > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to fine-tuned checkpoint')
    parser.add_argument('--corpus_file', required=True)
    parser.add_argument('--queries_file', required=True)
    parser.add_argument('--qrels_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--embeddings_cache', default=None)
    args = parser.parse_args()
    
    print("Loading fine-tuned ConstBERT model...")
    # Load base model first, then load fine-tuned weights
    from transformers import AutoModel
    import torch
    
    base_model = AutoModel.from_pretrained(
        'pinecone/ConstBERT',
        trust_remote_code=True
    )
    
    # Load fine-tuned weights (try both safetensors and pytorch_model.bin)
    checkpoint_path_safe = Path(args.model_path) / 'model.safetensors'
    checkpoint_path_pt = Path(args.model_path) / 'pytorch_model.bin'
    
    if checkpoint_path_safe.exists():
        print(f"Loading fine-tuned weights from {checkpoint_path_safe}...")
        from safetensors.torch import load_file
        state_dict = load_file(str(checkpoint_path_safe))
        base_model.load_state_dict(state_dict, strict=False)
        print("✅ Fine-tuned weights loaded successfully")
    elif checkpoint_path_pt.exists():
        print(f"Loading fine-tuned weights from {checkpoint_path_pt}...")
        state_dict = torch.load(checkpoint_path_pt, map_location='cpu')
        base_model.load_state_dict(state_dict, strict=False)
        print("✅ Fine-tuned weights loaded successfully")
    else:
        print("⚠️  No fine-tuned weights found, using base model")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_model.to(device)
    base_model.eval()
    
    # Create a minimal wrapper for encoding
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            self.device = device
            self.batch_size = 128
        
        def encode_queries(self, queries, **kwargs):
            with torch.no_grad():
                return self.model.encode_queries(queries, **kwargs)
        
        def encode_documents(self, documents, **kwargs):
            with torch.no_grad():
                return self.model.encode_documents(documents, **kwargs)
    
    model = ModelWrapper(base_model)
    from transformers import AutoModel
    import torch
    
    base_model = AutoModel.from_pretrained(
        'pinecone/ConstBERT',
        trust_remote_code=True
    )
    
    # Load fine-tuned weights
    checkpoint_path = Path(args.model_path) / 'pytorch_model.bin'
    if checkpoint_path.exists():
        print(f"Loading fine-tuned weights from {checkpoint_path}...")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        base_model.load_state_dict(state_dict)
    else:
        print("⚠️  No fine-tuned weights found, using base model")
    
    # Wrap in our wrapper for evaluation
    from models.constbert_wrapper import ConstBERTWrapper
    model = ConstBERTWrapper.__new__(ConstBERTWrapper)
    model.model = base_model
    model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.model.to(model.device)
    model.model.eval()
    model.batch_size = 128
    
    # Get tokenizers from model
    model.tokenizer = base_model.query_tokenizer
    
    print("Loading queries and qrels...")
    queries = load_queries(args.queries_file)
    qrels = load_qrels(args.qrels_file)
    
    print(f"Queries: {len(queries)}, Qrels: {len(qrels)}")
    
    # Load or create embeddings
    if args.embeddings_cache and Path(args.embeddings_cache).exists():
        print(f"Loading cached embeddings from {args.embeddings_cache}...")
        data = np.load(args.embeddings_cache, allow_pickle=True)
        doc_embeddings = data['embeddings']
        doc_ids = data['doc_ids'].tolist()
    else:
        print("Encoding corpus...")
        # Load corpus
        corpus_docs = []
        doc_ids = []
        with open(args.corpus_file) as f:
            for line in tqdm(f, desc="Loading corpus"):
                doc = json.loads(line)
                doc_ids.append(str(doc['id']))
                text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                corpus_docs.append(text)
        
        doc_embeddings = model.encode_documents(corpus_docs)
        
        if args.embeddings_cache:
            print(f"Saving embeddings to {args.embeddings_cache}...")
            np.savez_compressed(
                args.embeddings_cache,
                embeddings=doc_embeddings,
                doc_ids=np.array(doc_ids)
            )
    
    print("Building FAISS index...")
    index = ConstBERTFAISSIndex(embedding_dim=128, C=32)
    index.add_documents(doc_embeddings, doc_ids)
    
    print("Encoding queries and searching...")
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    query_embeddings = model.encode_queries(query_texts)
    
    rankings = {}
    for i, qid in enumerate(tqdm(query_ids, desc="Searching")):
        results = index.search(query_embeddings[i], k=1000)
        rankings[qid] = [doc_id for doc_id, _ in results]
    
    # Compute metrics
    mrr = compute_mrr(rankings, qrels)
    
    # Compute Recall@1000
    recall_sum = 0.0
    for qid in rankings:
        if qid in qrels:
            retrieved = set(rankings[qid][:1000])
            relevant = set(qrels[qid])
            recall_sum += len(retrieved & relevant) / len(relevant)
    
    recall_1000 = recall_sum / len(qrels) if qrels else 0.0
    
    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"  MRR@10: {mrr*100:.2f}%")
    print(f"  Recall@1000: {recall_1000*100:.2f}%")
    print(f"{'='*80}")
    
    # Save results
    results = {
        'mrr@10': mrr,
        'recall@1000': recall_1000,
        'num_queries': len(rankings)
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
EVAL_EOF
    chmod +x "$EVAL_SCRIPT"
fi

# Step 4: Evaluate on TEST split (final results for paper)
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║ STEP: Evaluate Fine-tuned Model on TEST Split (Final Results)"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

run_step "Evaluate on TEST split" python -u "$EVAL_SCRIPT" \
    --model_path "$OUTPUT_DIR" \
    --corpus_file "$CORPUS_FILE" \
    --queries_file "$TEST_QUERIES" \
    --qrels_file "$TEST_QRELS" \
    --output_file "${OUTPUT_DIR}/test_eval.json" \
    --embeddings_cache "${OUTPUT_DIR}/corpus_embeddings.npz"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║                           PIPELINE COMPLETED                                ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Results:"
echo "   Fine-tuned model: $OUTPUT_DIR"
echo "   TEST evaluation: ${OUTPUT_DIR}/test_eval.json (622 queries)"
echo "   Training log: ${OUTPUT_DIR}/training_log.json"
echo "   Full log: $LOG_FILE"
echo ""
echo "Data splits used:"
echo "   ✅ TRAIN (143 queries) → Fine-tuning"
echo "   ✅ TEST (622 queries) → Final evaluation (for paper)"
echo ""
echo "✅ All steps completed successfully!"
echo ""
echo "Next steps:"
echo "  1. Review results in ${OUTPUT_DIR}/test_eval.json"
echo "  2. Run same pipeline for ColBERT with identical splits"
echo "  3. Update paper with TEST split results"
