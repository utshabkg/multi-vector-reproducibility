"""
Experiment 6: Exact brute-force MaxSim evaluation.
Tests if PLAID approximation is causing the performance gap.
"""
import torch
import numpy as np
import time
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.constbert_wrapper import ConstBERTWrapper
from data.loaders import MSMARCODataLoader
from evaluation.metrics import compute_mrr, compute_recall

def main():
    # Load embeddings - keep on CPU, transfer chunks to GPU
    print('Loading document embeddings (memory-mapped)...')
    embeddings_path = 'indices/msmarco-passage/msmarco-passage-constbert/constbert_msmarco_embeddings.npy'
    doc_embs = np.load(embeddings_path, mmap_mode='r')
    print(f'Embeddings shape: {doc_embs.shape}, dtype={doc_embs.dtype}')
    
    # Load model
    constbert = ConstBERTWrapper(model_name='pinecone/ConstBERT', batch_size=1)
    
    # Load queries
    loader = MSMARCODataLoader()
    queries = loader.load_queries('dev')
    qrels = loader.load_qrels('dev')
    
    # Sample queries
    sample_size = 200
    query_ids = list(queries.keys())[:sample_size]
    print(f'Testing exact MaxSim on {len(query_ids)} queries...')
    
    results = {}
    k = 1000
    
    start = time.time()
    for qid in tqdm(query_ids, desc='Brute-force MaxSim'):
        Q = constbert.encode_queries([queries[qid]])
        Q = torch.from_numpy(Q).half().cuda().squeeze(0)  # (32, 128)
        
        # Batch MaxSim in chunks - load from CPU to GPU
        scores = []
        chunk_size = 200000  # Smaller chunks
        
        with torch.no_grad():
            for i in range(0, len(doc_embs), chunk_size):
                # Load chunk from mmap and transfer to GPU
                chunk_np = np.array(doc_embs[i:i+chunk_size])  # Make writable copy
                D_chunk = torch.from_numpy(chunk_np).half().cuda()  # (chunk, 32, 128)
                
                sim = torch.einsum('qd,bcd->bcq', Q, D_chunk)  # (chunk, 32, 32)
                max_sims = sim.max(dim=1).values  # (chunk, 32)
                chunk_scores = max_sims.sum(dim=1)  # (chunk,)
                scores.append(chunk_scores.cpu())
                
                del D_chunk, sim, max_sims
        
        all_scores = torch.cat(scores)
        top_k = torch.topk(all_scores, k)
        
        results[qid] = [(str(pid.item()), float(score.item())) 
                        for pid, score in zip(top_k.indices, top_k.values)]
    
    elapsed = time.time() - start
    print(f'\nCompleted in {elapsed:.1f}s ({len(query_ids)/elapsed:.2f} queries/sec)')
    
    mrr10 = compute_mrr(results, qrels, k=10)
    recall1000 = compute_recall(results, qrels, k=1000)
    
    print(f'\n=== Exact MaxSim Results (sample {sample_size}) ===')
    print(f'MRR@10:      {mrr10:.4f} ({mrr10*100:.2f}%)')
    print(f'Recall@1000: {recall1000:.4f} ({recall1000*100:.2f}%)')
    print(f'\nPLAID ncells=16: MRR@10=31.09%, Recall@1000=93.88%')
    print(f'Paper claims: MRR@10=39.04%, Recall@1000=96.34%')


if __name__ == '__main__':
    main()
