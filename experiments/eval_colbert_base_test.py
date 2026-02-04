#!/usr/bin/env python3
"""
Evaluate pretrained ColBERT-v2 (base) on ToT TEST split.
This provides a fair baseline comparison for the fine-tuned models.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add ColBERT to path
sys.path.insert(0, str(Path(__file__).parent.parent / "external/ColBERT"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
COLBERT_ROOT = PROJECT_ROOT / "colbert-reproduce-copy/colbert-replicability/colbert"
INDEX_PATH = Path("/media/12TB/shared/datasets/indices/trec-tot-2025/trec-tot-2025-colbertv2")
CHECKPOINT = Path("/media/12TB/shared/models/colbertv2.0")
TEST_QUERIES = Path("/media/12TB/shared/datasets/raw/trec-tot-2025/queries/test-2025-queries.jsonl")
TEST_QRELS = Path("/media/12TB/shared/datasets/raw/trec-tot-2025/qrel/test-2025-qrel.txt")
INDICES_ROOT = Path("/media/12TB/shared/datasets/indices/trec-tot-2025")
COLLECTION_TSV = COLBERT_ROOT / "tmp/trec-tot-2025-colbertv2_collection.tsv"

def main():
    from colbert import Searcher
    from colbert.infra import Run, RunConfig, ColBERTConfig
    
    logger.info("=" * 60)
    logger.info("Evaluating pretrained ColBERT-v2 (base) on ToT TEST")
    logger.info("=" * 60)
    
    # Load queries
    queries = {}
    with open(TEST_QUERIES) as f:
        for line in f:
            doc = json.loads(line)
            queries[str(doc['query_id'])] = doc['query']
    logger.info(f"Loaded {len(queries)} TEST queries")
    
    # Load qrels
    qrels = {}
    with open(TEST_QRELS) as f:
        for line in f:
            parts = line.strip().split()
            qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = rel
    logger.info(f"Loaded {len(qrels)} qrels")
    
    # Initialize searcher
    logger.info(f"Loading index from {INDEX_PATH}")
    with Run().context(RunConfig(nranks=1, experiment="eval_base_test")):
        config = ColBERTConfig(root=str(INDICES_ROOT))
        searcher = Searcher(
            index=str(INDEX_PATH.name),
            index_root=str(INDICES_ROOT),
            checkpoint=str(CHECKPOINT),
            config=config
        )
        
        # Load ID mapping (use full mapping derived from corpus)
        idmap_path = COLBERT_ROOT / f"tmp/{INDEX_PATH.name}_idmap_full.tsv"
        if not idmap_path.exists():
            # Fallback to original mapping
            idmap_path = COLBERT_ROOT / f"tmp/{INDEX_PATH.name}_idmap.tsv"
        internal_to_original = {}
        if idmap_path.exists():
            with open(idmap_path) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        internal_to_original[int(parts[0])] = parts[1]
            logger.info(f"Loaded ID mapping with {len(internal_to_original)} entries from {idmap_path.name}")
        
        # Search
        logger.info("Running search...")
        results = {}
        query_ids = list(queries.keys())
        
        for i, qid in enumerate(query_ids):
            query_text = queries[qid]
            ranking = searcher.search(query_text, k=1000)
            
            results[qid] = []
            for rank, (doc_id, score) in enumerate(zip(ranking[0], ranking[2])):
                if internal_to_original:
                    orig_id = internal_to_original.get(doc_id, str(doc_id))
                else:
                    orig_id = str(doc_id)
                results[qid].append((orig_id, float(score)))
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i+1}/{len(query_ids)} queries")
        
        logger.info(f"Search complete: {len(results)} queries")
    
    # Calculate metrics
    mrr_sum = 0
    recall_1000_sum = 0
    ndcg_10_sum = 0
    evaluated = 0
    
    for qid in results:
        if qid not in qrels:
            continue
        
        evaluated += 1
        rel_docs = {d for d, r in qrels[qid].items() if r > 0}
        
        # MRR@10
        for rank, (doc_id, score) in enumerate(results[qid][:10]):
            if doc_id in rel_docs:
                mrr_sum += 1.0 / (rank + 1)
                break
        
        # Recall@1000
        retrieved = {d for d, s in results[qid][:1000]}
        recall_1000_sum += len(retrieved & rel_docs) / len(rel_docs) if rel_docs else 0
        
        # NDCG@10
        dcg = 0
        for rank, (doc_id, score) in enumerate(results[qid][:10]):
            rel = qrels[qid].get(doc_id, 0)
            dcg += (2**rel - 1) / (rank + 2)  # log2(rank+2)
        
        ideal_rels = sorted([r for r in qrels[qid].values() if r > 0], reverse=True)[:10]
        idcg = sum((2**r - 1) / (i + 2) for i, r in enumerate(ideal_rels))
        ndcg_10_sum += dcg / idcg if idcg > 0 else 0
    
    metrics = {
        "MRR@10": mrr_sum / evaluated,
        "Recall@1000": recall_1000_sum / evaluated,
        "NDCG@10": ndcg_10_sum / evaluated
    }
    
    logger.info("")
    logger.info("=" * 40)
    logger.info("RESULTS: ColBERT-v2 (pretrained) on TEST")
    logger.info("=" * 40)
    logger.info(f"  MRR@10: {metrics['MRR@10']*100:.2f}%")
    logger.info(f"  Recall@1000: {metrics['Recall@1000']*100:.2f}%")
    logger.info(f"  NDCG@10: {metrics['NDCG@10']*100:.2f}%")
    
    # Save results
    output = {
        "model": "colbert-v2-pretrained",
        "split": "TEST",
        "queries": len(queries),
        "evaluated": evaluated,
        "metrics": {
            "MRR@10": metrics["MRR@10"],
            "Recall@1000": metrics["Recall@1000"],
            "NDCG@10": metrics["NDCG@10"]
        }
    }
    
    results_path = PROJECT_ROOT / "results/colbert_base_test.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
