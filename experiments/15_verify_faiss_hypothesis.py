#!/usr/bin/env python3
"""
Verify FAISS corpus-scale hypothesis.

Hypothesis: FEVER/Quora FAISS degradation is due to insufficient nlist 
(1024 clusters for 5.4M docs = too sparse for ConstBERT's 32 vectors).

Test: Run FEVER with nlist=4096 and nlist=8192 to see if performance recovers.
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.beir_evaluation import (
    BEIRDataset, 
    encode_beir_corpus, 
    encode_beir_queries,
    OptimizedFAISSIndex,
    retrieve_and_evaluate
)
from models.constbert_wrapper import ConstBERTWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_experiment(dataset_name: str, nlist: int, nprobe: int = 128):
    """Run BEIR evaluation with specific nlist parameter."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Dataset: {dataset_name}, nlist={nlist}, nprobe={nprobe}")
    logger.info(f"{'='*80}\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = ConstBERTWrapper(device=device)
    
    # Load dataset
    data_dir = Path(__file__).parent.parent / 'data' / 'beir'
    beir_dataset = BEIRDataset(dataset_name, str(data_dir))
    beir_dataset.download()
    corpus, queries, qrels = beir_dataset.load()
    
    logger.info(f"Corpus size: {len(corpus):,} documents")
    logger.info(f"Queries: {len(queries):,}")
    
    # Encode corpus
    start_time = time.time()
    doc_embeddings, doc_ids = encode_beir_corpus(
        model, corpus, batch_size=1024, num_workers=40
    )
    encode_time = time.time() - start_time
    logger.info(f"Document encoding: {encode_time:.2f}s")
    
    # Build FAISS index with specified nlist
    start_time = time.time()
    faiss_index = OptimizedFAISSIndex(use_gpu_maxsim=True)
    faiss_index.build_index(doc_embeddings, doc_ids, nlist=nlist, nprobe=nprobe)
    index_time = time.time() - start_time
    logger.info(f"Index building: {index_time:.2f}s")
    
    # Encode queries
    start_time = time.time()
    query_embeddings, query_ids = encode_beir_queries(
        model, queries, batch_size=512
    )
    query_time = time.time() - start_time
    logger.info(f"Query encoding: {query_time:.2f}s")
    
    # Retrieve and evaluate
    start_time = time.time()
    metrics = retrieve_and_evaluate(
        faiss_index, query_embeddings, query_ids, qrels,
        k=1000, batch_size=512
    )
    retrieval_time = time.time() - start_time
    logger.info(f"Retrieval: {retrieval_time:.2f}s")
    
    return {
        'dataset': dataset_name,
        'nlist': nlist,
        'nprobe': nprobe,
        'num_docs': len(corpus),
        'metrics': metrics,
        'timing': {
            'encode_time': encode_time,
            'index_time': index_time,
            'query_time': query_time,
            'retrieval_time': retrieval_time
        }
    }


def main():
    results = {}
    
    # Test FEVER with different nlist values
    for nlist in [1024, 4096, 8192]:
        logger.info(f"\n\nTesting FEVER with nlist={nlist}...")
        result = run_experiment('fever', nlist=nlist)
        results[f'fever_nlist{nlist}'] = result
        
        # Save intermediate result
        output_path = Path(__file__).parent.parent / 'results' / f'15_fever_nlist{nlist}.json'
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved: {output_path}")
        
        # Print key metrics
        print(f"\n=== FEVER nlist={nlist} ===")
        print(f"NDCG@10: {result['metrics']['ndcg@10']*100:.2f}%")
        print(f"MRR@10:  {result['metrics']['mrr@10']*100:.2f}%")
        print(f"R@1000:  {result['metrics']['recall@1000']*100:.2f}%")
    
    # Summary comparison
    print("\n\n" + "="*80)
    print("SUMMARY: FEVER nlist Sensitivity")
    print("="*80)
    print(f"{'nlist':<10} {'NDCG@10':<15} {'MRR@10':<15} {'R@1000':<15}")
    print("-"*55)
    
    for key in ['fever_nlist1024', 'fever_nlist4096', 'fever_nlist8192']:
        r = results[key]
        nlist = r['nlist']
        ndcg = r['metrics']['ndcg@10'] * 100
        mrr = r['metrics']['mrr@10'] * 100
        recall = r['metrics']['recall@1000'] * 100
        print(f"{nlist:<10} {ndcg:<15.2f} {mrr:<15.2f} {recall:<15.2f}")
    
    # Compare with PLAID reference
    print("\nReference: PLAID (32K centroids) = 57.1% NDCG@10")
    print("\nHypothesis: If nlist=4096+ shows significant improvement,")
    print("corpus-scale sensitivity is confirmed.")
    
    # Save full summary
    summary_path = Path(__file__).parent.parent / 'results' / '15_nlist_hypothesis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {summary_path}")


if __name__ == '__main__':
    main()
