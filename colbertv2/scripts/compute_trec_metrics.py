#!/usr/bin/env python3
import sys
import json
import math
from collections import defaultdict


def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            qid = parts[0]
            docid = parts[2]
            rel = int(parts[3])
            qrels[qid][docid] = rel
    return qrels


def read_run(path):
    runs = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            qid = parts[0]
            docid = parts[2]
            runs[qid].append(docid)
    return runs


def dcg(rels):
    s = 0.0
    for i, r in enumerate(rels, start=1):
        s += (2**r - 1.0) / math.log2(i + 1)
    return s


def compute_metrics(qrels, runs):
    metrics = {'map': [], 'recip_rank': [], 'ndcg_cut_10': [], 'recall_1000': []}

    for qid, qrels_docs in qrels.items():
        rels_in_qrels = sum(1 for v in qrels_docs.values() if v > 0)
        run_list = runs.get(qid, [])
        run_rels = [qrels_docs.get(d, 0) for d in run_list[:1000]]

        # MRR
        rr = 0.0
        for i, r in enumerate(run_rels, start=1):
            if r > 0:
                rr = 1.0 / i
                break
        metrics['recip_rank'].append(rr)

        # AP
        if rels_in_qrels == 0:
            ap = 0.0
        else:
            num_rel_found = 0
            sum_prec = 0.0
            for i, r in enumerate(run_rels, start=1):
                if r > 0:
                    num_rel_found += 1
                    sum_prec += num_rel_found / i
            ap = sum_prec / rels_in_qrels if rels_in_qrels > 0 else 0.0
        metrics['map'].append(ap)

        # NDCG@10
        k = 10
        rels_k = run_rels[:k] + [0] * max(0, k - len(run_rels))
        dcg_k = dcg(rels_k)
        ideal_rels = sorted([v for v in qrels_docs.values()], reverse=True)[:k]
        idcg_k = dcg(ideal_rels + [0] * max(0, k - len(ideal_rels)))
        ndcg = (dcg_k / idcg_k) if idcg_k > 0 else 0.0
        metrics['ndcg_cut_10'].append(ndcg)

        # Recall@1000
        if rels_in_qrels == 0:
            rec = 0.0
        else:
            rec = sum(1 for r in run_rels if r > 0) / rels_in_qrels
        metrics['recall_1000'].append(rec)

    # aggregate means
    agg = {}
    for k, vals in metrics.items():
        agg[k] = sum(vals) / len(vals) if vals else 0.0
    return agg


def main():
    if len(sys.argv) < 4:
        print('Usage: compute_trec_metrics.py <qrels> <trec_run> <out_json> [queries_jsonl]')
        sys.exit(2)
    qrels = read_qrels(sys.argv[1])
    runs = read_run(sys.argv[2])

    # Optional: remap numeric run qids using original queries JSONL
    if len(sys.argv) >= 5 and sys.argv[4]:
        qjson = sys.argv[4]
        idx_to_qid = {}
        try:
            import json as _json
            with open(qjson) as fq:
                for i, line in enumerate(fq):
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = _json.loads(line)
                    except Exception:
                        # fallback: assume TSV qid\ttext
                        parts = line.split('\t')
                        if parts:
                            idx_to_qid[i] = parts[0]
                            continue
                    # find an id field
                    for key in ('qid','id','query_id','QID'):
                        if key in obj:
                            idx_to_qid[i] = str(obj[key])
                            break
                    else:
                        # no id field; use first key's value as fallback
                        if isinstance(obj, dict) and obj:
                            first = next(iter(obj.values()))
                            idx_to_qid[i] = str(first)

            # remap runs
            new_runs = defaultdict(list)
            for qid, docs in runs.items():
                mapped = qid
                if qid.isdigit():
                    qi = int(qid)
                    if qi in idx_to_qid:
                        mapped = idx_to_qid[qi]
                new_runs[mapped].extend(docs)
            runs = new_runs
        except Exception:
            pass

    out = compute_metrics(qrels, runs)
    with open(sys.argv[3], 'w') as fo:
        json.dump(out, fo, indent=2)
    print('WROTE', sys.argv[3])


if __name__ == '__main__':
    main()
