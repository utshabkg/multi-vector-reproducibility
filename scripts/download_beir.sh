#!/bin/bash
# Download BEIR benchmark datasets
# Source: https://github.com/beir-cellar/beir
set -e

DATA_DIR="${1:-data/beir}"
mkdir -p "$DATA_DIR"

echo "Downloading BEIR datasets to $DATA_DIR..."

# BEIR datasets used in the paper (RQ3)
DATASETS=(
    "nfcorpus"
    "fiqa"
    "arguana"
    "scidocs"
    "scifact"
    "trec-covid"
    "webis-touche2020"
    "nq"
    "hotpotqa"
    "fever"
    "climate-fever"
    "dbpedia-entity"
    "quora"
)

BASE_URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

for dataset in "${DATASETS[@]}"; do
    if [ -d "$DATA_DIR/$dataset" ]; then
        echo "$dataset already exists, skipping."
    else
        echo "Downloading $dataset..."
        wget -P "$DATA_DIR" "$BASE_URL/$dataset.zip"
        unzip -q "$DATA_DIR/$dataset.zip" -d "$DATA_DIR"
        rm "$DATA_DIR/$dataset.zip"
    fi
done

echo "BEIR datasets download complete."
echo "Downloaded datasets:"
ls -d "$DATA_DIR"/*/
