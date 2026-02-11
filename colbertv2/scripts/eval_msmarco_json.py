#!/usr/bin/env python3
import os, ujson, tqdm
from collections import defaultdict
import sys

def compute_metrics(qrels_path, ranking_path):
    qid2positives = defaultdict(list)
    qid2ranking = defaultdict(list)

    with open(qrels_path) as f:
        for line in f:
            qid, _, pid, label = map(int, line.strip().split())
            if label == 1:
                qid2positives[qid].append(pid)

    with open(ranking_path) as f:
        for line in f:
            qid, pid, rank, *score = line.strip().split('\t')
            qid, pid, rank = int(qid), int(pid), int(rank)
            qid2ranking[qid].append((rank, pid))

    num_judged = len(qid2positives)
    num_ranked = len(qid2ranking)

    qid2mrr = {}
    recalls = {d: {} for d in [50,200,1000,5000,10000]}

    for qid, positives in qid2positives.items():
        ranking = qid2ranking.get(qid, [])
        for idx, (_, pid) in enumerate(ranking, start=1):
            if pid in positives:
                if idx <= 10:
                    qid2mrr[qid] = 1.0 / idx
                break
        for idx, (_, pid) in enumerate(ranking, start=1):
            if pid in positives:
                for d in recalls:
                    if idx <= d:
                        recalls[d][qid] = recalls[d].get(qid,0)+1.0/len(positives)

    mrr = sum(qid2mrr.values())/num_judged if num_judged>0 else 0.0
    metrics = {'mrr@10': mrr}
    for d in recalls:
        metrics[f'recall@{d}'] = sum(recalls[d].values())/num_judged if num_judged>0 else 0.0
    metrics['num_judged_queries'] = num_judged
    metrics['num_ranked_queries'] = num_ranked
    return metrics


def read_paper_claim(md_path):
    if not os.path.exists(md_path):
        return None
    with open(md_path) as f:
        text = f.read()
    # look for pattern like 0.360
    import re
    m = re.search(r"MRR@10\s*=\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"Expected\":\s*MRR@10\s*=\s*([0-9]*\.?[0-9]+)", text)
    if m:
        return float(m.group(1))
    # fallback: search for '0.360' explicitly
    m = re.search(r"0\.36[0-9]*", text)
    if m:
        return float(m.group(0))
    return None

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: eval_msmarco_json.py <qrels> <ranking> [out.json]')
        sys.exit(2)
    qrels = sys.argv[1]
    ranking = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else 'replicability/colbert/results/msmarco_eval_metrics.json'

    metrics = compute_metrics(qrels, ranking)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        ujson.dump(metrics, f, indent=2)
    print(f'Wrote metrics to {out}')

    # compare to paper claim from repo docs
    claim = read_paper_claim('REPRODUCTION_GAP_ANALYSIS.md')
    if claim is None:
        claim = read_paper_claim('IMPLEMENTATION_PLAN_9WEEKS.md')
    if claim is not None:
        diff = metrics['mrr@10'] - claim
        print(f"Paper claim MRR@10={claim:.4f}; obtained={metrics['mrr@10']:.4f}; diff={diff:+.4f}")
    else:
        print('Paper claim not found in local docs.')
