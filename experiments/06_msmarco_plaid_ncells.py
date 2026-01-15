"""
Experiment 5: PLAID evaluation with expanded ncells=16
Testing if larger ncells fixes the poor recall issue.
"""
import torch
import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.constbert_wrapper import ConstBERTWrapper
from colbert.infra import ColBERTConfig
from colbert.search.index_storage import IndexScorer
from data.loaders import MSMARCODataLoader
from evaluation.metrics import compute_mrr, compute_recall
from tqdm import tqdm

def main():
    index_path = '/media/12TB/shared/datasets/indices/msmarco-passage/constbert_plaid_index'
    
    # Use expanded parameters
    config = ColBERTConfig(
        ncells=16,  # Increased from 4
        ndocs=8192,  # Increased from 4096  
        centroid_score_threshold=0.3,  # Slightly lower
        index_path=index_path,
        checkpoint=index_path
    )
    
    print('Loading resources...')
    ranker = IndexScorer(index_path, use_gpu=True)
    constbert = ConstBERTWrapper(model_name='pinecone/ConstBERT', batch_size=128)
    
    # Load data
    loader = MSMARCODataLoader()
    queries = loader.load_queries('dev')
    qrels = loader.load_qrels('dev')
    
    print(f'Loaded {len(queries)} queries')
    print(f'Using ncells={config.ncells}, ndocs={config.ndocs}, threshold={config.centroid_score_threshold}')
    
    # Run evaluation
    k = 1000
    results = {}
    query_ids = list(queries.keys())
    batch_size = 128
    
    start_time = time.time()
    
    for i in tqdm(range(0, len(query_ids), batch_size), desc='Evaluating'):
        batch_qids = query_ids[i:i+batch_size]
        batch_queries = [queries[qid] for qid in batch_qids]
        
        Q = constbert.encode_queries(batch_queries)
        Q_torch = torch.from_numpy(Q).cuda().float()
        
        for j, qid in enumerate(batch_qids):
            pids, scores = ranker.rank(config, Q_torch[j:j+1])
            
            # Handle output format
            if isinstance(pids, list) and pids and isinstance(pids[0], int):
                top_pids = pids[:k]
                top_scores = scores[:k]
            else:
                top_pids = pids[0].tolist() if hasattr(pids[0], 'tolist') else pids[0]
                top_scores = scores[0].tolist() if hasattr(scores[0], 'tolist') else scores[0]
                top_pids = top_pids[:k]
                top_scores = top_scores[:k]
            
            # Convert to expected format: doc_ids are strings in qrels
            results[qid] = [(str(pid), float(score)) for pid, score in zip(top_pids, top_scores)]
    
    elapsed = time.time() - start_time
    print(f'\nCompleted in {elapsed:.1f}s ({len(results)/elapsed:.1f} queries/sec)')
    
    # Compute metrics
    mrr10 = compute_mrr(results, qrels, k=10)
    recall1000 = compute_recall(results, qrels, k=1000)
    
    print(f'\n=== Results with ncells=16 ===')
    print(f'MRR@10:      {mrr10:.4f} ({mrr10*100:.2f}%)')
    print(f'Recall@1000: {recall1000:.4f} ({recall1000*100:.2f}%)')
    print(f'\nPaper claims: MRR@10=39.04%, Recall@1000=96.34%')
    print(f'Previous (ncells=4): MRR@10=30.01%, Recall@1000=87.64%')
    
    # Save results
    results_path = Path(__file__).parent.parent / 'results' / 'exp5_ncells16_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'mrr@10': mrr10,
            'recall@1000': recall1000,
            'config': {
                'ncells': config.ncells,
                'ndocs': config.ndocs,
                'threshold': config.centroid_score_threshold
            },
            'num_queries': len(results),
            'time_seconds': elapsed
        }, f, indent=2)
    print(f'\nSaved to {results_path}')


if __name__ == '__main__':
    main()
