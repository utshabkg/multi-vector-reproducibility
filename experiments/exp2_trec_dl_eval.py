"""
Experiment 2: TREC Deep Learning Track 2019 and 2020 Evaluation
Reproducing Table 1 TREC DL results from ConstBERT paper.

Reuses existing embeddings and index from Experiment 1.
"""
import sys
sys.path.append('/home/ugdf8/IRIS/dev/reproduce/constbert-reproduce')

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from data.loaders import MSMARCODataLoader
from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import evaluate_retrieval

def load_trec_queries(year):
    """Load TREC DL queries."""
    data_dir = Path("/media/12TB/shared/datasets/raw/msmarco-passage")
    query_path = data_dir / f"msmarco-test{year}-queries.tsv"
    
    queries = {}
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                query = parts[1]
                queries[qid] = query
    
    return queries

def load_trec_qrels(year):
    """Load TREC DL qrels (graded relevance 0-3)."""
    data_dir = Path("/media/12TB/shared/datasets/raw/msmarco-passage")
    qrel_path = data_dir / f"{year}qrels-pass.txt"
    
    qrels = {}
    with open(qrel_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                
                if qid not in qrels:
                    qrels[qid] = {}
                qrels[qid][doc_id] = relevance
    
    return qrels

def main():
    print("=" * 80)
    print("ConstBERT Reproducibility: TREC Deep Learning Track Evaluation")
    print("=" * 80)
    
    # Load model once (shared between years)
    print(f"\n[1/4] Initializing ConstBERT model...")
    model = ConstBERTWrapper(batch_size=32)
    
    # Load index once (shared between years)
    print(f"\n[2/4] Loading pre-built FAISS index...")
    print("This may take 2-3 minutes (loading 272 GB from disk)...")
    index_path = "/media/12TB/shared/datasets/indices/constbert_msmarco_faiss_index.pkl"
    load_start = time.time()
    index = ConstBERTFAISSIndex.load(index_path)
    load_time = time.time() - load_start
    print(f"Index loaded in {load_time:.1f} seconds: {len(index.doc_ids)} documents")
    
    # Process both years
    for year in [2019, 2020]:
        print(f"\n{'=' * 80}")
        print(f"TREC DL {year} Evaluation")
        print(f"{'=' * 80}")
        
        # Load queries and qrels
        print(f"\n[3/4] Loading TREC DL {year} data...")
        queries = load_trec_queries(year)
        qrels = load_trec_qrels(year)
        
        # Filter to queries with qrels
        query_ids = [qid for qid in queries.keys() if qid in qrels]
        query_texts = [queries[qid] for qid in query_ids]
        
        print(f"Loaded {len(query_texts)} queries with qrels")
        
        # Encode queries
        print(f"\n[4/4] Encoding TREC DL {year} queries...")
        query_embeddings = model.encode_queries(query_texts, show_progress=True)
        print(f"Query embeddings shape: {query_embeddings.shape}")
        
        # Run retrieval
        print(f"\nRunning retrieval for {len(query_texts)} queries...")
        print(f"Using FAISS IVF + exact MaxSim (candidate_mult=10 for high accuracy)...")
        print(f"Estimated time: ~{len(query_texts) * 1.7:.0f} seconds ({len(query_texts) * 1.7 / 60:.1f} minutes)\n")
        
        start_time = time.time()
        all_doc_ids, all_scores = index.search(
            query_embeddings,
            k=1000,
            candidate_mult=10,  # Higher for TREC (fewer queries)
            show_progress=True
        )
        retrieval_time = time.time() - start_time
        
        print(f"\nRetrieval completed in {retrieval_time:.1f} seconds")
        print(f"Mean Response Time (MRT): {(retrieval_time / len(query_texts)) * 1000:.2f} ms")
        
        # Format results
        results = {}
        for qid, doc_ids, scores in zip(query_ids, all_doc_ids, all_scores):
            results[qid] = {doc_id: float(score) for doc_id, score in zip(doc_ids, scores)}
        
        # Evaluate
        print(f"\nComputing evaluation metrics...")
        # For TREC DL, we want NDCG@10 (primary), plus other standard metrics
        metrics_to_compute = ['ndcg@10', 'map@1000', 'recall@100', 'recall@1000', 'p@10']
        metrics = evaluate_retrieval(results, qrels, metrics=metrics_to_compute)
        
        # Display results
        print(f"\n{'=' * 60}")
        print(f"              TREC DL {year} Results")
        print(f"{'=' * 60}")
        for metric, value in sorted(metrics.items()):
            print(f"{metric:<20}: {value:.4f}")
        print(f"{'=' * 60}")
        
        # Compare with paper
        paper_results = {
            2019: {"ndcg@10": 0.7314, "map@1000": None},
            2020: {"ndcg@10": 0.7329, "map@1000": None}
        }
        
        if year in paper_results:
            print(f"\nComparison with Paper (ConstBERT32):")
            print(f"{'Metric':<20} {'Ours':<15} {'Paper':<15} {'Diff':<15}")
            print("-" * 60)
            
            for metric_key in ["ndcg@10"]:
                if metric_key in metrics and paper_results[year][metric_key]:
                    ours = metrics[metric_key] * 100
                    paper = paper_results[year][metric_key] * 100
                    diff = ours - paper
                    print(f"{metric_key:<20} {ours:>13.2f}% {paper:>13.2f}% {diff:>+13.2f}%")
        
        # Save results
        output_file = f"results/exp2_trec_dl{year}_results.json"
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "year": year,
            "num_queries": len(query_texts),
            "metrics": metrics,
            "mean_response_time_ms": (retrieval_time / len(query_texts)) * 1000,
            "paper_comparison": {}
        }
        
        if year in paper_results and "ndcg@10" in metrics:
            output_data["paper_comparison"]["ndcg@10"] = {
                "ours": metrics["ndcg@10"] * 100,
                "paper": paper_results[year]["ndcg@10"] * 100 if paper_results[year]["ndcg@10"] else None,
                "diff": (metrics["ndcg@10"] - paper_results[year]["ndcg@10"]) * 100 if paper_results[year]["ndcg@10"] else None
            }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    print(f"\n{'=' * 80}")
    print("Experiment complete!")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    main()
