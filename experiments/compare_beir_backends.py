#!/usr/bin/env python3
"""
Compare BEIR results between PLAID and FAISS backends.
Generates comparison tables for the paper.
"""

import json
from pathlib import Path

BEIR_DATASETS = [
    'arguana', 'climate-fever', 'dbpedia-entity', 'fever', 'fiqa',
    'hotpotqa', 'nfcorpus', 'nq', 'quora', 'scidocs', 'scifact',
    'trec-covid', 'webis-touche2020'
]

DATASET_NAMES = {
    'arguana': 'ArguAna',
    'climate-fever': 'Climate-FEVER',
    'dbpedia-entity': 'DBPedia',
    'fever': 'FEVER',
    'fiqa': 'FiQA-2018',
    'hotpotqa': 'HotpotQA',
    'nfcorpus': 'NFCorpus',
    'nq': 'NQ',
    'quora': 'Quora',
    'scidocs': 'SCIDOCS',
    'scifact': 'SciFact',
    'trec-covid': 'TREC-COVID',
    'webis-touche2020': 'Touché-2020'
}

def main():
    # Load FAISS results - try summary first, then individual files
    faiss_results = {}
    
    # Try summary file first
    faiss_summary_file = Path('results/11_beir_constbert_summary.json')
    if faiss_summary_file.exists():
        with open(faiss_summary_file) as f:
            faiss_results = json.load(f)
    else:
        # Load individual files
        for dataset in BEIR_DATASETS:
            faiss_file = Path(f'results/11_beir_constbert_{dataset}.json')
            if faiss_file.exists():
                with open(faiss_file) as f:
                    faiss_results[dataset] = json.load(f)

    # Load PLAID results - try summary first, then individual files
    plaid_results = {}
    
    plaid_summary_file = Path('results/12_beir_plaid_summary.json')
    if plaid_summary_file.exists():
        with open(plaid_summary_file) as f:
            plaid_results = json.load(f)
    else:
        # Load individual files
        for dataset in BEIR_DATASETS:
            plaid_file = Path(f'results/12_beir_plaid_{dataset}.json')
            if plaid_file.exists():
                with open(plaid_file) as f:
                    plaid_results[dataset] = json.load(f)

    print("\n" + "="*100)
    print("BEIR Backend Comparison: PLAID vs FAISS (All 13 Datasets)")
    print("="*100 + "\n")

    # NDCG@10 comparison
    print("Table 1: NDCG@10 Comparison")
    print("-" * 100)
    print(f"{'Dataset':<20} {'PLAID':>10} {'FAISS':>10} {'Difference':>12} {'Paper':>10} {'Status':>10}")
    print("-" * 100)

    plaid_ndcgs = []
    faiss_ndcgs = []
    
    # Paper values from ConstBERT paper Table 2 (only 5 datasets reported)
    paper_values = {
        'scifact': 0.607,
        'nfcorpus': 0.327,
        'fiqa': 0.312,
        'arguana': 0.451,
        'trec-covid': 0.745,
        'climate-fever': 0.142,
        'dbpedia-entity': 0.418,
        'fever': 0.696,
        'hotpotqa': 0.621,
        'nq': 0.534,
        'quora': 0.821,
        'scidocs': 0.156,
        'webis-touche2020': 0.260,
    }

    for dataset in BEIR_DATASETS:
        if dataset not in plaid_results and dataset not in faiss_results:
            continue
        
        # Get results with proper handling of missing data
        plaid_ndcg = None
        faiss_ndcg = None
        
        if dataset in plaid_results:
            plaid_ndcg = plaid_results[dataset].get('ndcg@10') or plaid_results[dataset].get('metrics', {}).get('ndcg@10')
        
        if dataset in faiss_results:
            faiss_ndcg = faiss_results[dataset].get('ndcg@10') or faiss_results[dataset].get('metrics', {}).get('ndcg@10')
        
        # Skip if both are missing
        if plaid_ndcg is None and faiss_ndcg is None:
            continue
            
        paper_ndcg = paper_values.get(dataset)

        # Calculate difference if both exist
        diff = None
        if plaid_ndcg is not None and faiss_ndcg is not None:
            diff = plaid_ndcg - faiss_ndcg
            plaid_ndcgs.append(plaid_ndcg)
            faiss_ndcgs.append(faiss_ndcg)

        # Format output
        plaid_str = f"{plaid_ndcg*100:>9.2f}%" if plaid_ndcg is not None else "     ---"
        faiss_str = f"{faiss_ndcg*100:>9.2f}%" if faiss_ndcg is not None else "     ---"
        diff_str = f"{diff*100:>11.2f}%" if diff is not None else "        ---"
        paper_str = f"{paper_ndcg*100:>9.2f}%" if paper_ndcg else "     ---"
        
        # Status indicator
        if paper_ndcg and faiss_ndcg is not None:
            faiss_vs_paper = abs(faiss_ndcg - paper_ndcg)
            if faiss_vs_paper < 0.011:  # Within 1.1%
                status = "✓"
            elif faiss_vs_paper < 0.05:  # Within 5%
                status = "~"
            else:
                status = "✗"
        else:
            status = "---"

        print(f"{DATASET_NAMES[dataset]:<20} {plaid_str} {faiss_str} "
              f"{diff_str} {paper_str} {status:>10}")

    mean_plaid = sum(plaid_ndcgs) / len(plaid_ndcgs)
    mean_faiss = sum(faiss_ndcgs) / len(faiss_ndcgs)
    mean_diff = mean_plaid - mean_faiss
    
    # Mean for 5 reported datasets only
    reported_datasets = ['scifact', 'nfcorpus', 'fiqa', 'arguana', 'trec-covid']
    mean_paper = sum(v for k, v in paper_values.items() if v is not None) / len([v for v in paper_values.values() if v is not None])

    print("-" * 100)
    print(f"{'Mean (all)':<20} {mean_plaid*100:>9.2f}% {mean_faiss*100:>9.2f}% "
          f"{mean_diff*100:>11.2f}% {mean_paper*100:>9.2f}%")
    print("="*100 + "\n")

    # Degradation analysis
    print("Performance Degradation Analysis")
    print("-" * 100)
    print(f"{'Dataset':<20} {'PLAID vs Paper':>16} {'FAISS vs Paper':>16} {'PLAID vs FAISS':>17}")
    print("-" * 100)

    for dataset in BEIR_DATASETS:
        if dataset not in plaid_results and dataset not in faiss_results:
            continue
        
        # Get results with proper handling
        plaid_ndcg = None
        faiss_ndcg = None
        
        if dataset in plaid_results:
            plaid_ndcg = plaid_results[dataset].get('ndcg@10') or plaid_results[dataset].get('metrics', {}).get('ndcg@10')
        
        if dataset in faiss_results:
            faiss_ndcg = faiss_results[dataset].get('ndcg@10') or faiss_results[dataset].get('metrics', {}).get('ndcg@10')
        
        if plaid_ndcg is None and faiss_ndcg is None:
            continue
            
        paper_ndcg = paper_values.get(dataset)

        plaid_vs_faiss = None
        if plaid_ndcg is not None and faiss_ndcg is not None:
            plaid_vs_faiss = (plaid_ndcg - faiss_ndcg) / faiss_ndcg * 100

        if paper_ndcg:
            plaid_vs_paper = (plaid_ndcg - paper_ndcg) / paper_ndcg * 100 if plaid_ndcg else None
            faiss_vs_paper = (faiss_ndcg - paper_ndcg) / paper_ndcg * 100 if faiss_ndcg else None
            
            plaid_str = f"{plaid_vs_paper:>15.1f}%" if plaid_vs_paper is not None else "           ---"
            faiss_str = f"{faiss_vs_paper:>15.1f}%" if faiss_vs_paper is not None else "           ---"
            plaid_faiss_str = f"{plaid_vs_faiss:>16.1f}%" if plaid_vs_faiss is not None else "            ---"
            
            print(f"{DATASET_NAMES[dataset]:<20} {plaid_str} {faiss_str} {plaid_faiss_str}")
        else:
            plaid_faiss_str = f"{plaid_vs_faiss:>16.1f}%" if plaid_vs_faiss is not None else "            ---"
            print(f"{DATASET_NAMES[dataset]:<20} {'           ---':>15} {'           ---':>15} {plaid_faiss_str}")

    mean_plaid_vs_paper = (mean_plaid - mean_paper) / mean_paper * 100
    mean_faiss_vs_paper = (mean_faiss - mean_paper) / mean_paper * 100
    mean_plaid_vs_faiss = (mean_plaid - mean_faiss) / mean_faiss * 100

    print("-" * 100)
    print(f"{'Mean (reported)':<20} {mean_plaid_vs_paper:>15.1f}% {mean_faiss_vs_paper:>15.1f}% {mean_plaid_vs_faiss:>16.1f}%")
    print("="*100 + "\n")

    # Summary statistics
    print("Summary")
    print("-" * 100)
    print(f"Total datasets evaluated: {len(plaid_ndcgs)}")
    print(f"Datasets in paper: {len([v for v in paper_values.values() if v is not None])}")
    print()
    print(f"PLAID mean NDCG@10 (all):       {mean_plaid*100:.2f}%")
    print(f"FAISS mean NDCG@10 (all):       {mean_faiss*100:.2f}%")
    print(f"Paper mean NDCG@10 (5 datasets): {mean_paper*100:.2f}%")
    print()
    print(f"PLAID degradation vs paper:     {mean_plaid_vs_paper:+.1f}%")
    print(f"FAISS difference vs paper:      {mean_faiss_vs_paper:+.1f}%")
    print(f"PLAID degradation vs FAISS:     {mean_plaid_vs_faiss:+.1f}%")
    print("="*100 + "\n")

    # LaTeX table for paper
    print("LaTeX Table (for paper - all datasets)")
    print("-" * 100)
    print(r"\begin{tabular}{lrrr}")
    print(r"\toprule")
    print(r"\textbf{Dataset} & \textbf{PLAID-16} & \textbf{FAISS} & \textbf{$\Delta$} \\")
    print(r"\midrule")

    for dataset in BEIR_DATASETS:
        if dataset not in plaid_results and dataset not in faiss_results:
            continue
        
        # Get results with proper handling
        plaid_ndcg = None
        faiss_ndcg = None
        
        if dataset in plaid_results:
            plaid_ndcg = plaid_results[dataset].get('ndcg@10') or plaid_results[dataset].get('metrics', {}).get('ndcg@10')
        
        if dataset in faiss_results:
            faiss_ndcg = faiss_results[dataset].get('ndcg@10') or faiss_results[dataset].get('metrics', {}).get('ndcg@10')
        
        if plaid_ndcg is None and faiss_ndcg is None:
            continue
            
        diff = None
        if plaid_ndcg is not None and faiss_ndcg is not None:
            diff = plaid_ndcg - faiss_ndcg

        # Bold negative difference (degradation)
        if diff is not None:
            diff_str = f"\\textbf{{{diff*100:+.1f}\\%}}" if diff < -0.03 else f"{diff*100:+.1f}\\%"
        else:
            diff_str = "---"
        
        plaid_str = f"{plaid_ndcg*100:.1f}\\%" if plaid_ndcg is not None else "---"
        faiss_str = f"{faiss_ndcg*100:.1f}\\%" if faiss_ndcg is not None else "---"

        print(f"{DATASET_NAMES[dataset]} & {plaid_str} & {faiss_str} & {diff_str} \\\\")

    print(r"\midrule")
    diff_str = f"\\textbf{{{mean_diff*100:+.1f}\\%}}"
    print(f"\\textbf{{Mean}} & \\textbf{{{mean_plaid*100:.1f}\\%}} & \\textbf{{{mean_faiss*100:.1f}\\%}} & {diff_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print("="*100 + "\n")

    # Save comparison
    comparison = {
        'plaid': {dataset: plaid_results[dataset]['ndcg@10'] for dataset in BEIR_DATASETS if dataset in plaid_results},
        'faiss': {dataset: faiss_results[dataset]['metrics']['ndcg@10'] for dataset in BEIR_DATASETS if dataset in faiss_results},
        'paper': {k: v for k, v in paper_values.items() if v is not None},
        'mean': {
            'plaid': mean_plaid,
            'faiss': mean_faiss,
            'paper': mean_paper,
        },
        'degradation': {
            'plaid_vs_paper_pct': mean_plaid_vs_paper,
            'faiss_vs_paper_pct': mean_faiss_vs_paper,
            'plaid_vs_faiss_pct': mean_plaid_vs_faiss,
        }
    }

    comparison_file = Path('results/12_beir_backend_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"Comparison saved to: {comparison_file}\n")


if __name__ == "__main__":
    main()
