#!/usr/bin/env python3
"""Convert ranking file internal pids back to original pids using mapping.

Input ranking format: qid\tpid\tscore (or qid\tpid\t...); pid values are internal numeric ids.
Mapping format: internal_id\toriginal_pid per line.

Writes converted ranking to stdout or to --out path.
"""
import sys
from argparse import ArgumentParser


def load_map(map_path):
    m = {}
    with open(map_path) as f:
        for line in f:
            a, b = line.strip().split('\t')
            m[a] = b
    return m


def convert(ranking_in, map_path, out_path=None):
    m = load_map(map_path)
    out = open(out_path, 'w', encoding='utf-8') if out_path else sys.stdout
    with open(ranking_in) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2:
                continue
            qid = parts[0]
            pid = parts[1]
            rest = parts[2:]
            orig = m.get(pid)
            if orig is None:
                # If mapping not found, keep pid
                orig = pid
            out.write('\t'.join([qid, str(orig)] + rest) + '\n')
    if out_path:
        out.close()


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('ranking')
    p.add_argument('map')
    p.add_argument('--out', help='Output converted ranking path')
    args = p.parse_args()
    convert(args.ranking, args.map, args.out)
