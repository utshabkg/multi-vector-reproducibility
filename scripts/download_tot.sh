#!/bin/bash
# Download TREC Tip-of-the-Tongue (ToT) 2025 dataset
# Source: https://trec-tot.github.io/
set -e

DATA_DIR="${1:-data/trec-tot-2025}"
mkdir -p "$DATA_DIR/queries"
mkdir -p "$DATA_DIR/qrel"

echo "Downloading TREC ToT 2025 dataset to $DATA_DIR..."
echo ""
echo "NOTE: The TREC ToT 2025 dataset requires registration."
echo "Please download from: https://trec-tot.github.io/"
echo ""
echo "Expected files:"
echo "  $DATA_DIR/trec-tot-2025-corpus.jsonl     (document corpus)"
echo "  $DATA_DIR/queries/dev1-queries.jsonl      (dev split 1)"
echo "  $DATA_DIR/queries/dev2-queries.jsonl      (dev split 2)"
echo "  $DATA_DIR/queries/dev3-queries.jsonl      (dev split 3)"
echo "  $DATA_DIR/queries/test-2025-queries.jsonl (test queries)"
echo "  $DATA_DIR/qrel/dev1-qrel.txt              (dev1 relevance judgments)"
echo "  $DATA_DIR/qrel/dev2-qrel.txt              (dev2 relevance judgments)"
echo "  $DATA_DIR/qrel/dev3-qrel.txt              (dev3 relevance judgments)"
echo "  $DATA_DIR/qrel/test-2025-qrel.txt         (test relevance judgments)"
echo ""
echo "After downloading, place the files in the directories shown above."
