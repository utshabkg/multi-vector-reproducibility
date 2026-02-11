#!/usr/bin/env python3
"""
Generate training triples using an existing BM25 Pyserini index.

Writes lines of the form: qid\tpos_pid\tneg_pid

Defaults assume an existing BM25 index at
`indices/trec-tot-2025-bm25` and writes
triples to `data/processed/tot/train.triples`.
"""
import json
import os
import argparse
from collections import defaultdict
import re

try:
    import pyterrier as pt
    PT_AVAILABLE = True
    if not pt.started():
        try:
            pt.init()
        except Exception:
            pass
except Exception:
    PT_AVAILABLE = False

import pandas as pd

def load_qrels(path):
    qrels = defaultdict(set)
    with open(path, 'r') as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = parts[0]
            pid = parts[2]
            qrels[qid].add(pid)
    return qrels

def load_queries_jsonl(path):
    queries = {}
    with open(path, 'r') as fh:
        for line in fh:
            obj = json.loads(line)
            # try common keys
            qid = None
            for k in ('id','qid','query_id'):
                if k in obj:
                    qid = str(obj[k]); break
            if qid is None:
                # fallback to incremental id (not ideal)
                continue
            text = obj.get('text') or obj.get('query') or obj.get('query_text') or obj.get('q') or ''
            # sanitize queries to avoid Terrier parser lexical errors
            # remove special characters except word chars, whitespace, and hyphen
            text = re.sub(r"[^\w\s-]", ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            queries[qid] = text
    return queries

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default='indices/trec-tot-2025-bm25', help='Path to existing pyserini BM25 index')
    # Defaults point to the TRAIN split (do not use dev1 for training)
    parser.add_argument('--queries', default='data/trec-tot-2025/raw/queries/train-2025-queries.jsonl', help='Queries JSONL (train split)')
    parser.add_argument('--qrels', default='data/trec-tot-2025/raw/qrel/train-2025-qrel.txt', help='Qrels file (train split)')
    parser.add_argument('--out', default='data/processed/trec-tot-2025-triple-bm25/train.triples', help='Output triples path')
    parser.add_argument('--topk', type=int, default=100, help='Candidates to retrieve per query')
    parser.add_argument('--negs', type=int, default=8, help='Negatives per positive')
    args = parser.parse_args()

    # Prefer PyTerrier if available (works with PyTerrier index dir), otherwise fall back to Pyserini
    use_pyterrier = PT_AVAILABLE

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    qrels = load_qrels(args.qrels)
    queries = load_queries_jsonl(args.queries)

    # Try to initialise retrieval and an index reader to map internal docids -> stored corpus ids
    index_reader = None
    retriever = None
    searcher = None
    if use_pyterrier:
        try:
            # Try to create a BatchRetrieve pipeline against the provided index
            try:
                index_ref = pt.IndexRef.of(args.index)
            except Exception:
                index_ref = pt.IndexFactory.of(args.index)
            retriever = pt.BatchRetrieve(index_ref, wmodel='BM25')
        except Exception as e:
            raise SystemExit(f'Failed to initialise PyTerrier BatchRetrieve for index {args.index}: {e}')
    else:
        try:
            # Preferred modern import for pyserini
            from pyserini.search.lucene import LuceneSearcher as PyseriniSearcher
        except Exception:
            try:
                from pyserini.search import SimpleSearcher as PyseriniSearcher
            except Exception:
                raise SystemExit('pyserini is required: pip install pyserini')
        try:
            searcher = PyseriniSearcher(args.index)
        except Exception as e:
            msg = str(e)
            if 'IndexNotFoundException' in msg or 'no segments' in msg or 'no segments*' in msg:
                listing = os.listdir(args.index) if os.path.isdir(args.index) else 'not a directory'
                raise SystemExit(f"Lucene index not found at {args.index}. Directory listing: {listing}\n" \
                                  "Make sure the path points to a pyserini/lucene index (contains segments_N files),\n" \
                                  "or supply a correct index path via --index.")
            raise

    # Try to create an IndexReader (pyserini) if available so we can map docids -> stored id fields
    try:
        from pyserini.index import IndexReader
        try:
            index_reader = IndexReader(args.index)
        except Exception:
            try:
                index_reader = IndexReader.of(args.index)
            except Exception:
                index_reader = None
    except Exception:
        index_reader = None
    else:
        try:
            # Preferred modern import for pyserini
            from pyserini.search.lucene import LuceneSearcher as PyseriniSearcher
        except Exception:
            try:
                from pyserini.search import SimpleSearcher as PyseriniSearcher
            except Exception:
                raise SystemExit('pyserini is required: pip install pyserini')
        try:
            searcher = PyseriniSearcher(args.index)
        except Exception as e:
            msg = str(e)
            if 'IndexNotFoundException' in msg or 'no segments' in msg or 'no segments*' in msg:
                listing = os.listdir(args.index) if os.path.isdir(args.index) else 'not a directory'
                raise SystemExit(f"Lucene index not found at {args.index}. Directory listing: {listing}\n" \
                                  "Make sure the path points to a pyserini/lucene index (contains segments_N files),\n" \
                                  "or supply a correct index path via --index.")
            raise

    written = 0
    with open(args.out, 'w') as outfh:
        for qid, qtext in queries.items():
            if qid not in qrels:
                continue
            negs = []
            if use_pyterrier:
                qdf = pd.DataFrame([{'qid': qid, 'query': qtext}])
                try:
                    if hasattr(retriever, 'search'):
                        res = retriever.search(qdf)
                    else:
                        res = retriever.transform(qdf)
                except Exception:
                    # some BatchRetrieve pipelines are callable
                    try:
                        res = retriever(qdf)
                    except Exception as e:
                        raise SystemExit(f'PyTerrier retrieval failed: {e}')
                # res should be a DataFrame with a 'docid' or 'docno' column
                doccol = None
                for c in ('docno','docid','id'):
                    if c in res.columns:
                        doccol = c; break
                if doccol is None:
                    # fallback to first candidate column after qid/score
                    candidates = [c for c in res.columns if c not in ('qid','query','score')]
                    doccol = candidates[0] if candidates else 'docid'
                for pid in res[doccol].tolist():
                    # Map pid via index_reader if needed
                    mapped = None
                    try:
                        # if pid is an integer-like internal docid, try to fetch stored field
                        if index_reader is not None and str(pid).isdigit():
                            try:
                                doc = index_reader.doc(int(pid))
                                if isinstance(doc, dict):
                                    mapped = doc.get('id') or doc.get('docno') or doc.get('docid')
                            except Exception:
                                mapped = None
                        # if not mapped, assume pid already is the stored id
                        if mapped is None:
                            mapped = str(pid)
                    except Exception:
                        mapped = str(pid)
                    if mapped in qrels[qid]:
                        continue
                    negs.append(str(mapped))
                    if len(negs) >= args.negs:
                        break
            else:
                hits = searcher.search(qtext, k=args.topk)
                for h in hits:
                    # h.docid may be an internal id or a stored id; try to map using searcher/doc lookup
                    pid = h.docid
                    mapped = None
                    try:
                        if index_reader is not None and str(pid).isdigit():
                            try:
                                doc = index_reader.doc(int(pid))
                                if isinstance(doc, dict):
                                    mapped = doc.get('id') or doc.get('docno') or doc.get('docid')
                            except Exception:
                                mapped = None
                        if mapped is None:
                            # try searcher.doc if available
                            try:
                                d = searcher.doc(pid)
                                try:
                                    mapped = d.get('id') or d.get('docno') or d.get('docid')
                                except Exception:
                                    mapped = str(pid)
                            except Exception:
                                mapped = str(pid)
                    except Exception:
                        mapped = str(pid)
                    if mapped in qrels[qid]:
                        continue
                    negs.append(str(mapped))
                    if len(negs) >= args.negs:
                        break

            if not negs:
                continue
            for pos in qrels[qid]:
                for neg in negs:
                    outfh.write(f"{qid}\t{pos}\t{neg}\n")
                    written += 1

    print(f"Wrote {written} triples to {args.out}")

if __name__ == '__main__':
    main()
