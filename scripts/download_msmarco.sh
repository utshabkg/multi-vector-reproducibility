#!/bin/bash
# Download MS MARCO Passage Ranking dataset
# Source: https://microsoft.github.io/msmarco/
set -e

DATA_DIR="${1:-data/msmarco-passage}"
mkdir -p "$DATA_DIR"

echo "Downloading MS MARCO Passage Ranking dataset to $DATA_DIR..."

# Collection (8.8M passages)
if [ ! -f "$DATA_DIR/collection.tsv" ]; then
    echo "Downloading collection.tsv..."
    wget -P "$DATA_DIR" https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
    tar -xzf "$DATA_DIR/collection.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/collection.tar.gz"
else
    echo "collection.tsv already exists, skipping."
fi

# Dev queries (small)
if [ ! -f "$DATA_DIR/queries.dev.small.tsv" ]; then
    echo "Downloading queries.dev.small.tsv..."
    wget -P "$DATA_DIR" https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
    tar -xzf "$DATA_DIR/queries.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/queries.tar.gz"
else
    echo "queries.dev.small.tsv already exists, skipping."
fi

# Dev qrels (small)
if [ ! -f "$DATA_DIR/qrels.dev.small.tsv" ]; then
    echo "Downloading qrels.dev.small.tsv..."
    wget -P "$DATA_DIR" https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv
else
    echo "qrels.dev.small.tsv already exists, skipping."
fi

# TREC DL 2019
if [ ! -f "$DATA_DIR/2019qrels-pass.txt" ]; then
    echo "Downloading TREC DL 2019 qrels..."
    wget -O "$DATA_DIR/2019qrels-pass.txt" https://trec.nist.gov/data/deep/2019qrels-pass.txt
fi

# TREC DL 2020
if [ ! -f "$DATA_DIR/2020qrels-pass.txt" ]; then
    echo "Downloading TREC DL 2020 qrels..."
    wget -O "$DATA_DIR/2020qrels-pass.txt" https://trec.nist.gov/data/deep/2020qrels-pass.txt
fi

echo "MS MARCO Passage Ranking dataset download complete."
echo "Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
