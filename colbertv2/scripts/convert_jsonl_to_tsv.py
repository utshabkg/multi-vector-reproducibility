#!/usr/bin/env python3
"""Convert a JSONL collection to TSV (pid\ttext).

Expects each JSON line to be an object with keys: "id" or "pid" and "text" or "contents".
Writes to provided output path.
"""
import sys
import json
from argparse import ArgumentParser

def convert(inpath, outpath, map_outpath=None, use_original_pid=False):
    """Convert JSONL to TSV.

    By default, writes sequential numeric pids (0..N-1) to satisfy ColBERT's
    `load_collection` expectation. If `map_outpath` is provided, writes a TSV
    mapping file with lines: internal_id\toriginal_pid.

    If `use_original_pid` is True, preserve original pid values (legacy mode).
    """
    import os
    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    if map_outpath:
        os.makedirs(os.path.dirname(map_outpath) or '.', exist_ok=True)

    with open(inpath, 'r', encoding='utf-8') as fr, open(outpath, 'w', encoding='utf-8') as fw:
        line_idx = 0
        map_f = open(map_outpath, 'w', encoding='utf-8') if map_outpath else None

        for line in fr:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            orig_pid = str(obj.get('id') or obj.get('pid') or obj.get('doc_id') or obj.get('docno') or obj.get('query_id') or line_idx)
            text = obj.get('text') or obj.get('contents') or obj.get('body') or obj.get('passage') or obj.get('query')
            if text is None:
                line_idx += 1
                continue
            # sanitize tabs/newlines
            text = text.replace('\t', ' ').replace('\n', ' ')

            if use_original_pid:
                pid_to_write = orig_pid
            else:
                pid_to_write = str(line_idx)

            fw.write(f"{pid_to_write}\t{text}\n")

            if map_f is not None:
                map_f.write(f"{pid_to_write}\t{orig_pid}\n")

            line_idx += 1

        if map_f:
            map_f.close()

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('inpath')
    p.add_argument('outpath')
    p.add_argument('map_outpath', nargs='?', default=None, help='Optional mapping output path (internal_id\toriginal_pid)')
    p.add_argument('--use-original-pid', action='store_true', help='Keep original pid values instead of sequential ids')
    args = p.parse_args()
    convert(args.inpath, args.outpath, map_outpath=args.map_outpath, use_original_pid=args.use_original_pid)
