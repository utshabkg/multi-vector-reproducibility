"""
Experiment 1: MS-MARCO Dev Set Evaluation
Reproduce Table 1 results from the paper.
"""
import sys
import os
from pathlib import Path
import time
import json
import argparse
import numpy as np
from tqdm import tqdm

# Set HuggingFace cache to shared models directory
# os.environ['HF_HOME'] = 'path/to/hf_cache'  # Set if needed
# os.environ['TRANSFORMERS_CACHE'] = 'path/to/transformers_cache'  # Set if needed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.loaders import MSMARCODataLoader
from models.constbert_wrapper import ConstBERTWrapper
from models.faiss_index import ConstBERTFAISSIndex
from evaluation.metrics import evaluate_retrieval, print_evaluation_results


def main(args):
    """
    Run MS-MARCO dev set evaluation.
    
    Target metrics from paper (ConstBERT32):
    - MRR@10: 39.04
    - Recall@50: 85.86
    - Recall@200: 93.72
    - Recall@1000: 96.34
    """
    print("="*80)
    print("ConstBERT Reproducibility: MS-MARCO Dev Set Evaluation")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    data_loader = MSMARCODataLoader(args.data_dir)
    
    queries = data_loader.load_queries("dev")
    qrels = data_loader.load_qrels("dev")
    print(f"Loaded {len(queries)} queries, {len(qrels)} qrels")
    
    # 2. Initialize model
    print("\n[2/5] Initializing ConstBERT model...")
    model = ConstBERTWrapper(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    # 3. Build or load index
    # Use shared storage locations
    index_path = Path(args.index_dir) / "constbert_msmarco_faiss_index.pkl"
    embeddings_path = Path(args.embeddings_dir) / "constbert_msmarco_embeddings.npy"
    
    if args.load_index and index_path.exists():
        print(f"\n[3/5] Loading pre-built FAISS index from {index_path}...")
        index = ConstBERTFAISSIndex.load(index_path)
    else:
        print("\n[3/5] Building document index with FAISS...")
        print("This will take a while for 8.8M passages...")
        
        # Encode all documents
        if embeddings_path.exists() and args.load_index:
            print(f"Loading pre-computed embeddings from {embeddings_path}...")
            doc_embeddings = np.load(embeddings_path)
            
            # Load doc IDs
            collection = data_loader.load_collection()
            doc_ids = list(collection.keys())
        else:
            print("Encoding all passages (this may take hours)...")
            collection = data_loader.load_collection()
            doc_ids = list(collection.keys())
            doc_texts = [collection[doc_id] for doc_id in doc_ids]
            
            encode_start = time.time()
            doc_embeddings = model.encode_documents(doc_texts, show_progress=True)
            encode_time = time.time() - encode_start
            
            print(f"\nEncoding completed in {encode_time/3600:.2f} hours")
            print(f"Document embeddings shape: {doc_embeddings.shape}")
            
            # Save embeddings
            model.save_embeddings(doc_embeddings, embeddings_path)
        
        # Build FAISS index
        index = ConstBERTFAISSIndex(use_gpu=True)  # Use GPU for FAISS if available
        index.add_documents(doc_ids, doc_embeddings)
        
        # Save index
        index.save(index_path)
        
        # Report index size
        index_size_mb = index.get_index_size_mb()
        print(f"\nEmbeddings size: {index_size_mb:.2f} MB ({index_size_mb/1024:.2f} GB)")
        print(f"Expected from paper: ~11 GB for ConstBERT32")
    
    # 4. Run retrieval
    print("\n[4/5] Running retrieval on dev queries...")
    
    # Encode queries
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    print("Encoding queries...")
    query_embeddings = model.encode_queries(query_texts, show_progress=True)
    
    # Retrieve for each query
    print(f"\n[4/5] Running retrieval on dev queries...")
    print(f"Using FAISS for efficient candidate retrieval + exact MaxSim...")
    print(f"Strategy: FAISS finds {args.candidate_mult}x candidates, then exact MaxSim reranks")
    run = {}
    retrieval_times = []
    
    print(f"\nProcessing {len(query_ids)} queries...")
    start_retrieval = time.time()
    
    for i, qid in enumerate(tqdm(query_ids, desc="Retrieval", ncols=100)):
        start_time = time.time()
        doc_ids_list, scores_list = index.search(
            query_embeddings[i],
            k=args.top_k,
            candidate_mult=args.candidate_mult,
            show_progress=False  # Per-query progress disabled, using outer bar
        )
        retrieval_time = (time.time() - start_time) * 1000  # Convert to ms
        retrieval_times.append(retrieval_time)
        
        # Convert to format expected by evaluation: list of (doc_id, score) tuples
        run[qid] = [(doc_id, float(score)) for doc_id, score in zip(doc_ids_list[0], scores_list[0])]
        
        # Print stats every 100 queries
        if (i + 1) % 100 == 0:
            avg_time = np.mean(retrieval_times[-100:])
            elapsed = time.time() - start_retrieval
            queries_done = i + 1
            queries_remaining = len(query_ids) - queries_done
            eta_seconds = (elapsed / queries_done) * queries_remaining
            print(f"\n  [{queries_done}/{len(query_ids)}] Avg: {avg_time:.0f}ms/query | "
                  f"Elapsed: {elapsed/60:.1f}min | ETA: {eta_seconds/60:.1f}min")
    
    total_retrieval_time = time.time() - start_retrieval
    print(f"\nRetrieval completed in {total_retrieval_time/60:.1f} minutes")
    
    mean_response_time = sum(retrieval_times) / len(retrieval_times)
    print(f"\nMean Response Time (MRT): {mean_response_time:.2f} ms")
    print(f"Paper reports: ~51 ms for ColBERT (PLAID)")
    
    # 5. Evaluate
    print("\n[5/5] Computing evaluation metrics...")
    
    metrics_to_compute = [
        'mrr@10',
        'recall@50',
        'recall@200',
        'recall@1000',
    ]
    
    results = evaluate_retrieval(run, qrels, metrics_to_compute)
    
    # Print results
    print_evaluation_results(results, "MS-MARCO Dev Set Results")
    
    # Compare with paper
    paper_results = {
        'mrr@10': 39.04,
        'recall@50': 85.86,
        'recall@200': 93.72,
        'recall@1000': 96.34,
    }
    
    print("\nComparison with Paper (ConstBERT32):")
    print(f"{'Metric':<20} {'Ours':<12} {'Paper':<12} {'Diff':<12}")
    print("-" * 60)
    for metric in metrics_to_compute:
        ours = results[metric] * 100  # Convert to percentage
        paper = paper_results.get(metric, 0.0)
        diff = ours - paper
        print(f"{metric:<20} {ours:>10.2f}% {paper:>10.2f}% {diff:>+10.2f}%")
    
    # Save results
    output_path = Path(args.results_dir) / "exp1_dev_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': {k: float(v) for k, v in results.items()},
            'mean_response_time_ms': mean_response_time,
            'paper_comparison': {
                metric: {
                    'ours': results[metric] * 100,
                    'paper': paper_results[metric],
                    'diff': results[metric] * 100 - paper_results[metric]
                }
                for metric in metrics_to_compute
            }
        }, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nExperiment complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reproduce ConstBERT MS-MARCO dev set evaluation"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/msmarco-passage",
        help="Path to MS-MARCO dataset"
    )
    
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="data/msmarco-passage",
        help="Directory to save document embeddings"
    )
    
    parser.add_argument(
        "--index-dir",
        type=str,
        default="indices",
        help="Directory to save index"
    )
    
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="pinecone/ConstBERT",
        help="HuggingFace model name"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, None for auto)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of documents to retrieve"
    )
    
    parser.add_argument(
        "--candidate-mult",
        type=int,
        default=3,
        help="Candidate multiplier for FAISS (retrieve k*mult candidates for reranking)"
    )
    
    parser.add_argument(
        "--load-index",
        action="store_true",
        help="Load pre-built index if available"
    )
    
    args = parser.parse_args()
    main(args)
