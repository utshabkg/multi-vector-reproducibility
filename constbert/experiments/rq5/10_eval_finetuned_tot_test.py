"""
Experiment 4: TREC Tip-of-the-Tongue (ToT) 2025 - Fine-tuned Evaluation

Research Question: Does fine-tuning ConstBERT on ToT improve performance on 
descriptive, memory-based queries?

Fine-tuning Setup:
- Trained on: ToT TRAIN split (143 queries, 1,144 triples)
- Evaluated on: ToT TEST split (622 queries)

Dataset:
- Corpus: 6.4M Wikipedia articles
- Queries: 622 test queries (long, descriptive, tip-of-tongue style)
- Domain: Open Wikipedia vs MS-MARCO web passages
- Query style: "I remember..." narratives vs factoid/question queries

Comparison: Base ConstBERT vs Fine-tuned ConstBERT on ToT TEST split.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
import numpy as np
from tqdm import tqdm
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.tot_loader import TRECToTDataLoader
from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import (
    compute_mrr, compute_recall, compute_ndcg, compute_map
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run a fast test using a small subset of documents/queries")
    parser.add_argument("--splits", nargs='+', default=["test"],
                        choices=["test", "dev1", "dev2", "dev3"],
                        help="Which split(s) to evaluate: test and/or dev1/dev2/dev3")
    args = parser.parse_args()
    # Paths - all stored together in index directory
    index_dir = Path("indices/trec-tot-2025/constbert_tot_faiss_index")
    embeddings_path = index_dir / "embeddings.npy"
    metadata_path = index_dir / "metadata.npy"
    index_path = index_dir / "index"
    # Results filename depends on chosen split to avoid overwriting
    split = 'test' if not hasattr(args, 'split') else args.split
    results_path = project_root / "results" / f"10_finetune_tot_{split}.json"
    
    # Create directories
    index_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    logger.info("=" * 80)
    logger.info("Experiment 4: TREC Tip-of-the-Tongue (ToT) 2025 - Fine-tuned Evaluation")
    logger.info("=" * 80)
    
    # Load data
    logger.info("\n[1/5] Loading ToT dataset...")
    loader = TRECToTDataLoader()
    
    # Get corpus statistics
    stats = loader.get_corpus_statistics()
    logger.info(f"  Corpus: {stats['total_documents']:,} Wikipedia articles")
    logger.info(f"  Average passage length: {stats['avg_passage_length']:.0f} characters")
    
    # Splits to evaluate (we'll load queries per-split later)
    splits = args.splits
    logger.info(f"  Splits to evaluate: {splits}")
    
    # Check if embeddings exist
    if Path(embeddings_path).exists() and Path(metadata_path).exists():
        logger.info(f"\n[2/5] Loading pre-computed embeddings from {embeddings_path}...")
        # In test mode, memory-map the embeddings and only materialize a small slice
        if args.test:
            logger.info("  TEST MODE: using memory-mapped load and truncation to avoid full file I/O")
            embeddings_mem = np.load(embeddings_path, mmap_mode='r')
            doc_ids = np.load(metadata_path, allow_pickle=True)
            subset_docs = min(10000, embeddings_mem.shape[0])
            embeddings = np.array(embeddings_mem[:subset_docs])
            doc_ids = doc_ids[:subset_docs]
            logger.info(f"  Loaded embeddings (truncated): shape={embeddings.shape}, dtype={embeddings.dtype}")
            logger.info(f"  Documents (truncated): {len(doc_ids):,}")
        else:
            embeddings = np.load(embeddings_path)
            doc_ids = np.load(metadata_path, allow_pickle=True)
            logger.info(f"  Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
            logger.info(f"  Documents: {len(doc_ids):,}")
    else:
        logger.info("\n[2/5] Computing document embeddings...")
        logger.info("  Loading full corpus (this may take a while)...")
        doc_ids, passages = loader.load_corpus()
        logger.info(f"  Loaded {len(passages):,} passages")
        
        # Initialize model
        logger.info("  Loading FINE-TUNED ConstBERT model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"  Using device: {device}")
        
        # Load base model then fine-tuned weights
        from transformers import AutoModel
        from safetensors.torch import load_file
        
        base_model = AutoModel.from_pretrained('pinecone/ConstBERT', trust_remote_code=True)
        checkpoint_path = Path('checkpoints/constbert_tot_finetuned/model.safetensors')
        if checkpoint_path.exists():
            logger.info(f"  ✅ Loading fine-tuned weights from {checkpoint_path}")
            state_dict = load_file(str(checkpoint_path))
            base_model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning("  ⚠️  Fine-tuned weights not found!")
        
        # Wrap the model
        model = ConstBERTWrapper.__new__(ConstBERTWrapper)
        model.model = base_model
        model.device = device
        model.model.to(device)
        model.model.eval()
        model.batch_size = 256
        model.tokenizer = base_model.query_tokenizer
        
        # Encode documents
        logger.info(f"  Encoding documents (batch_size=256 - optimized for 48GB VRAM)...")
        start_time = time.time()
        embeddings = model.encode_documents(passages, batch_size=256, show_progress=True)
        encoding_time = time.time() - start_time
        
        logger.info(f"  Encoding completed in {encoding_time/60:.1f} minutes")
        logger.info(f"  Embeddings shape: {embeddings.shape}")
        logger.info(f"  Embeddings dtype: {embeddings.dtype}")
        
        # Save embeddings
        logger.info(f"  Saving embeddings to {embeddings_path}...")
        np.save(embeddings_path, embeddings)
        np.save(metadata_path, np.array(doc_ids))
        logger.info("  Embeddings saved!")
        
        # Calculate storage
        emb_size = Path(embeddings_path).stat().st_size / (1024**3)
        meta_size = Path(metadata_path).stat().st_size / (1024**3)
        logger.info(f"  Storage: {emb_size:.2f} GB (embeddings) + {meta_size:.2f} GB (metadata)")
    
    # Check if index exists or run test-mode in-memory index
    if args.test:
        logger.info("\n[3/5] TEST MODE: Building small in-memory FAISS index (no disk I/O)")
        # Force CPU FAISS for small test
        index = ConstBERTFAISSIndex(use_gpu=False)
        logger.info("  Adding documents to in-memory index...")
        start_time = time.time()
        index.add_documents(doc_ids, embeddings)
        build_time = time.time() - start_time
        logger.info(f"  In-memory index built in {build_time:.2f} seconds!")
        logger.info(f"  Index contains {index.num_docs:,} documents")
    else:
        if Path(index_path).exists() and Path(f"{index_path}.faiss").exists():
            logger.info(f"\n[3/5] Loading pre-built FAISS index from {index_path}...")
            logger.info("  This may take 2-3 minutes...")
            start_time = time.time()
            index = ConstBERTFAISSIndex.load(index_path)
            load_time = time.time() - start_time
            logger.info(f"  Index loaded in {load_time:.1f} seconds!")
            logger.info(f"  Index contains {index.num_docs:,} documents")
        else:
            logger.info("\n[3/5] Building FAISS IVF index...")
            logger.info("  This enables efficient retrieval on 6.4M documents")
            
            # Build index
            index = ConstBERTFAISSIndex(use_gpu=torch.cuda.is_available())
            
            logger.info("  Adding documents to index (with IVF training)...")
            start_time = time.time()
            index.add_documents(doc_ids, embeddings)
            build_time = time.time() - start_time
            logger.info(f"  Index built in {build_time/60:.1f} minutes!")
            
            # Save index
            logger.info(f"  Saving index to {index_path}...")
            index.save(index_path)
            logger.info("  Index saved!")
            
            # Calculate storage
            pkl_size = Path(index_path).stat().st_size / (1024**3)
            faiss_size = Path(f"{index_path}.faiss").stat().st_size / (1024**3)
            logger.info(f"  Storage: {pkl_size:.2f} GB (metadata) + {faiss_size:.2f} GB (FAISS index)")
    
    # Initialize FINE-TUNED model once for query encoding (used for all splits)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoModel
    from safetensors.torch import load_file

    base_model = AutoModel.from_pretrained('pinecone/ConstBERT', trust_remote_code=True)
    checkpoint_path = Path('checkpoints/constbert_tot_finetuned/model.safetensors')
    if checkpoint_path.exists():
        logger.info(f"  ✅ Loading fine-tuned weights from {checkpoint_path}")
        state_dict = load_file(str(checkpoint_path))
        base_model.load_state_dict(state_dict, strict=False)

    model = ConstBERTWrapper.__new__(ConstBERTWrapper)
    model.model = base_model
    model.device = device
    model.model.to(device)
    model.model.eval()
    model.batch_size = 128
    model.tokenizer = base_model.query_tokenizer

    # Evaluate each requested split (embeddings/index already loaded)
    for split in splits:
        logger.info(f"\n[4/5] Running retrieval for split='{split}' ({len(loader.load_queries(split))} queries)...")
        logger.info("  Using FAISS IVF with candidate_mult=10 for high accuracy")

        # Load queries/qrels for this split
        test_queries = loader.load_queries(split)
        test_qrels = loader.load_qrels(split)

        # Show sample query
        sample_qid = list(test_queries.keys())[0]
        sample_query = test_queries[sample_qid]
        logger.info(f"\n  Sample query (ID {sample_qid}):")
        logger.info(f"  '{sample_query[:150]}...'")
        logger.info(f"  Length: {len(sample_query)} chars")
        logger.info(f"  Relevant docs: {len(test_qrels[sample_qid])}")

        # Encode queries
        logger.info("  Encoding queries...")
        query_ids = list(test_queries.keys())
        query_texts = [test_queries[qid] for qid in query_ids]
        query_embeddings = model.encode_queries(query_texts, batch_size=128, show_progress=True)

        # Retrieve
        logger.info("  Retrieving top-1000 documents per query...")
        all_results = {}
        query_times = []

        for i, qid in enumerate(tqdm(query_ids, desc=f"Retrieving-{split}")):
            start_time = time.time()

            # Retrieve (returns doc_id lists and score lists)
            doc_lists, score_lists = index.search(
                query_embeddings[i:i+1],
                k=1000,
                candidate_mult=10  # Higher for better accuracy
            )
            results = list(zip(doc_lists[0], score_lists[0]))

            query_time = time.time() - start_time
            query_times.append(query_time)

            # Store results as {doc_id: score}
            all_results[qid] = {doc_id: float(score) for doc_id, score in results}
        
        # Compute metrics for this split
        logger.info("\n[5/5] Computing evaluation metrics...")

        # MRR@10
        mrr_10 = compute_mrr(all_results, test_qrels, k=10)
        logger.info(f"  MRR@10: {mrr_10*100:.2f}%")

        # Recall@k
        recall_50 = compute_recall(all_results, test_qrels, k=50)
        recall_200 = compute_recall(all_results, test_qrels, k=200)
        recall_1000 = compute_recall(all_results, test_qrels, k=1000)
        logger.info(f"  Recall@50: {recall_50*100:.2f}%")
        logger.info(f"  Recall@200: {recall_200*100:.2f}%")
        logger.info(f"  Recall@1000: {recall_1000*100:.2f}%")

        # NDCG@10
        ndcg_10 = compute_ndcg(all_results, test_qrels, k=10)
        logger.info(f"  NDCG@10: {ndcg_10*100:.2f}%")

        # MAP@1000
        map_1000 = compute_map(all_results, test_qrels, k=1000)
        logger.info(f"  MAP@1000: {map_1000*100:.2f}%")

        # Mean response time
        mean_rt = np.mean(query_times) * 1000  # ms
        logger.info(f"  Mean Response Time: {mean_rt:.0f} ms")

        # Save results (split-specific filename)
        split_results_path = project_root / "results" / f"finetune-results_tot_{split}.json"
        split_results_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "experiment": f"ToT Fine-tuned Evaluation (split={split})",
            "dataset": {
                "corpus_size": stats['total_documents'],
                "num_queries": len(test_queries),
                "avg_passage_length": stats['avg_passage_length']
            },
            "model": {
                "name": "pinecone/ConstBERT (fine-tuned on ToT TRAIN)",
                "base_model": "pinecone/ConstBERT",
                "fine_tuned": True,
                "training_data": "ToT TRAIN split (143 queries, 1,144 triples)",
                "C": 32,
                "embedding_dim": 128
            },
            "retrieval": {
                "method": "FAISS IVF + MaxSim",
                "nlist": 4096,
                "nprobe": 128,
                "candidate_mult": 10,
                "k": 1000
            },
            "metrics": {
                "MRR@10": float(mrr_10),
                "Recall@50": float(recall_50),
                "Recall@200": float(recall_200),
                "Recall@1000": float(recall_1000),
                "NDCG@10": float(ndcg_10),
                "MAP@1000": float(map_1000),
                "Mean_Response_Time_ms": float(mean_rt)
            },
            "sample_queries": [
                {
                    "query_id": sample_qid,
                    "query": test_queries[sample_qid][:200] + "...",
                    "relevant_docs": len(test_qrels[sample_qid]),
                    "top_3_results": [
                        {"doc_id": doc_id, "score": float(score)}
                        for doc_id, score in list(all_results[sample_qid].items())[:3]
                    ]
                }
            ]
        }

        with open(split_results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to {split_results_path}")
        logger.info(f"{'='*80}")

        # Summary
        logger.info("\n=== SPLIT COMPLETE ===")
        logger.info(f"ToT {split.upper()} Set ({len(test_queries)} queries):")
        logger.info(f"  MRR@10: {mrr_10*100:.2f}%")
        logger.info(f"  Recall@1000: {recall_1000*100:.2f}%")
        logger.info(f"  NDCG@10: {ndcg_10*100:.2f}%")
        logger.info(f"  Mean Response Time: {mean_rt:.0f} ms")
        logger.info("\nFine-tuned model will be compared with base ConstBERT for improvement analysis.")
    

    

if __name__ == "__main__":
    main()
