"""
Storage analysis for ConstBERT (C=32) on MS MARCO passage.

Paper numbers this script backs:
  * Sec 4 Author Communication: "our float16 embeddings occupy 67.5GB".
    The embedding matrix is num_docs x 32 x 128 float16 values. With
    8,841,823 passages that is 72,432,214,016 bytes, which is 67.46 GiB
    (72.43 GB in decimal units). The paper writes the GiB value as GB.
  * Sec 4 Datasets: "MS-MARCO Passage (8.8M passages)". The passage count is
    read from the shape of the embedding matrix, not typed in.
  * Sec 4 Author Communication: the 11GB in the ConstBERT paper refers to a
    PLAID compressed index. This script also reports the size of our own
    ConstBERT PLAID index so the two can be compared.

The ConstBERT paper values (11 GB for ConstBERT C=32 and 22 GB for its ColBERT
baseline) are cited constants from that paper. They are not our results.

Earlier versions of this script crashed before printing anything (Path was
used before it was imported) and pointed at relative paths that do not exist.
This version reads the real files on shared disk and writes a JSON record.

Usage (from the repo root, CPU only, reads .npy headers only):
  python constbert/experiments/rq1/03_storage_analysis.py
  python constbert/experiments/rq1/03_storage_analysis.py --output constbert/results/03_storage_analysis.json
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

MSMARCO_DIR = "/media/12TB/shared/datasets/indices/msmarco-passage"

# Cited from the ConstBERT paper (index sizes on MS MARCO passage).
PAPER_CONSTBERT32_SIZE_GB = 11
PAPER_COLBERT_SIZE_GB = 22


def gib(n):
    return None if n is None else n / 1024 ** 3


def gb(n):
    return None if n is None else n / 1e9


def stub(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    st = p.stat()
    mtime = dt.datetime.fromtimestamp(st.st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if p.is_dir():
        return {"path": str(p), "exists": True, "is_dir": True, "du_sb_bytes": du_sb(p), "mtime_utc": mtime}
    return {"path": str(p), "exists": True, "size_bytes": st.st_size, "mtime_utc": mtime}


def du_sb(path):
    try:
        out = subprocess.run(["du", "-sb", str(path)], capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError):
        return None


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_status(path):
    rel = os.path.relpath(Path(path).resolve(), REPO_ROOT)
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--", rel],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or "clean"
    except (OSError, subprocess.CalledProcessError):
        return None


def conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env and env != "base":
        return env
    prefix = Path(sys.prefix)
    return prefix.name if prefix.parent.name == "envs" else (env or str(prefix))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="ConstBERT MS MARCO storage analysis")
    ap.add_argument("--embeddings", default=f"{MSMARCO_DIR}/msmarco-passage-constbert/constbert_msmarco_embeddings.npy")
    ap.add_argument("--faiss-index", default=f"{MSMARCO_DIR}/msmarco-passage-constbert/constbert_msmarco_faiss_index.faiss")
    ap.add_argument("--faiss-pkl", default=f"{MSMARCO_DIR}/msmarco-passage-constbert/constbert_msmarco_faiss_index.pkl")
    ap.add_argument("--plaid-index", default=f"{MSMARCO_DIR}/constbert_plaid_index")
    ap.add_argument("--output", default=str(REPO_ROOT / "constbert/results/03_storage_analysis.json"))
    args = ap.parse_args()

    embeddings_path = Path(args.embeddings)
    index_faiss_path = Path(args.faiss_index)
    index_pkl_path = Path(args.faiss_pkl)
    plaid_dir = Path(args.plaid_index)

    print("=" * 80)
    print("ConstBERT Storage Analysis")
    print("=" * 80)

    if not embeddings_path.exists():
        sys.exit(f"Embeddings not found: {embeddings_path}")
    emb = np.load(embeddings_path, mmap_mode="r")
    NUM_DOCS, C, DIM = emb.shape

    print("\n1. THEORETICAL STORAGE CALCULATION")
    print("-" * 80)
    print(f"Number of documents: {NUM_DOCS:,}")
    print(f"Vectors per document (C): {C}")
    print(f"Embedding dimension: {DIM}")
    theory = {}
    for name, nbytes in (("float32", 4), ("float16", 2), ("int8", 1)):
        per_doc = C * DIM * nbytes
        total = NUM_DOCS * per_doc
        theory[name] = {"bytes_per_value": nbytes, "bytes_per_doc": per_doc, "total_bytes": total,
                        "total_gib": round(gib(total), 3), "total_gb_decimal": round(gb(total), 3)}
        print(f"\nWith {name} ({nbytes} bytes):")
        print(f"  Bytes per document: {per_doc:,}")
        print(f"  Total storage: {gib(total):.2f} GiB ({gb(total):.2f} GB decimal)")

    print("\n2. ACTUAL STORAGE ON DISK")
    print("-" * 80)
    emb_size = embeddings_path.stat().st_size
    print(f"Embeddings (.npy): {gib(emb_size):.2f} GiB ({emb_size:,} bytes)")
    print(f"  Shape: {emb.shape}")
    print(f"  Dtype: {emb.dtype}")
    print(f"  Data bytes: {emb.nbytes:,} ({gib(emb.nbytes):.2f} GiB)")
    faiss_size = index_faiss_path.stat().st_size if index_faiss_path.exists() else None
    pkl_size = index_pkl_path.stat().st_size if index_pkl_path.exists() else None
    if faiss_size is not None:
        print(f"\nFAISS index (.faiss): {gib(faiss_size):.2f} GiB")
    if pkl_size is not None:
        print(f"Metadata (.pkl, holds doc ids and a copy of the embeddings): {gib(pkl_size):.2f} GiB")
    faiss_total = sum(x for x in (emb_size, faiss_size, pkl_size) if x is not None)
    print(f"\nTotal FAISS-IVF setup on disk: {gib(faiss_total):.2f} GiB")
    plaid_size = du_sb(plaid_dir) if plaid_dir.exists() else None
    plaid_meta = {}
    if (plaid_dir / "metadata.json").exists():
        with open(plaid_dir / "metadata.json") as f:
            plaid_meta = json.load(f)
    if plaid_size is not None:
        print(f"PLAID index ({plaid_dir.name}, nbits={plaid_meta.get('config', {}).get('nbits')}): "
              f"{gib(plaid_size):.2f} GiB ({gb(plaid_size):.2f} GB decimal)")

    print("\n3. COMPARISON WITH PAPER")
    print("-" * 80)
    print(f"ConstBERT paper (ConstBERT32 PLAID index): {PAPER_CONSTBERT32_SIZE_GB} GB")
    print(f"Our float16 embeddings: {gib(emb.nbytes):.2f} GiB, ratio {gib(emb.nbytes) / PAPER_CONSTBERT32_SIZE_GB:.1f}x")
    print(f"Our FAISS-IVF setup: {gib(faiss_total):.2f} GiB, ratio {gib(faiss_total) / PAPER_CONSTBERT32_SIZE_GB:.1f}x")
    if plaid_size is not None:
        print(f"Our PLAID index: {gb(plaid_size):.2f} GB, ratio {gb(plaid_size) / PAPER_CONSTBERT32_SIZE_GB:.2f}x")

    print("\n4. STORAGE BREAKDOWN")
    print("-" * 80)
    print(f"  Raw embeddings ({emb.dtype}): {gib(emb_size):.1f} GiB")
    if faiss_size is not None:
        print(f"  FAISS IVF index (float32 vectors plus ids): {gib(faiss_size):.1f} GiB")
    if pkl_size is not None:
        print(f"  Pickle (doc ids plus a second copy of the embeddings): {gib(pkl_size):.1f} GiB")

    print("\n" + "=" * 80)

    result = {
        "description": "Storage of ConstBERT MS MARCO embeddings and indices, compared with the ConstBERT paper.",
        "paper_claims": {
            "this_paper_sec4_fp16_embeddings": "our float16 embeddings occupy 67.5GB",
            "this_paper_sec4_msmarco_passages": "8.8M passages",
            "constbert_paper_constbert32_index_gb": PAPER_CONSTBERT32_SIZE_GB,
            "constbert_paper_colbert_index_gb": PAPER_COLBERT_SIZE_GB,
        },
        "embedding_matrix": {"num_docs": int(NUM_DOCS), "vectors_per_doc": int(C), "dim": int(DIM),
                             "dtype": str(emb.dtype), "data_bytes": int(emb.nbytes),
                             "data_gib": round(gib(emb.nbytes), 3), "data_gb_decimal": round(gb(emb.nbytes), 3),
                             "npy_file_bytes": emb_size, "npy_file_gib": round(gib(emb_size), 3)},
        "theoretical": theory,
        "on_disk": {
            "faiss_index_bytes": faiss_size, "faiss_index_gib": round(gib(faiss_size), 3) if faiss_size else None,
            "faiss_pkl_bytes": pkl_size, "faiss_pkl_gib": round(gib(pkl_size), 3) if pkl_size else None,
            "faiss_setup_total_bytes": faiss_total, "faiss_setup_total_gib": round(gib(faiss_total), 3),
            "plaid_index_bytes": plaid_size,
            "plaid_index_gib": round(gib(plaid_size), 3) if plaid_size else None,
            "plaid_index_gb_decimal": round(gb(plaid_size), 3) if plaid_size else None,
            "plaid_nbits": plaid_meta.get("config", {}).get("nbits"),
            "plaid_num_partitions": plaid_meta.get("num_partitions"),
        },
        "checks": {
            "fp16_embeddings_gib_rounded_1dp": round(gib(emb.nbytes), 1),
            "matches_paper_67_5": round(gib(emb.nbytes), 1) == 67.5,
            "unit_note": "67.5 is the GiB value. In decimal GB the matrix is about 72.4 GB.",
            "num_docs_millions_rounded_1dp": round(NUM_DOCS / 1e6, 1),
        },
        "provenance": {
            "script": os.path.relpath(Path(__file__).resolve(), REPO_ROOT),
            "script_sha256": sha256(Path(__file__).resolve()),
            "command": shlex.join([sys.executable] + sys.argv),
            "conda_env": conda_env(),
            "git_head": git_head(),
            "script_git_status_at_run_time": git_status(__file__),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": socket.gethostname(),
            "random_seeds": None,
            "inputs": {"embeddings_npy": stub(embeddings_path), "faiss_index": stub(index_faiss_path),
                       "faiss_pkl": stub(index_pkl_path), "plaid_index_dir": stub(plaid_dir),
                       "plaid_metadata_json": stub(plaid_dir / "metadata.json")},
            "notes": ["Large files are not hashed. Path, size and mtime are recorded instead.",
                      "The .npy file is memory mapped, so only its header is read."],
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
