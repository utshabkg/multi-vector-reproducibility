#!/usr/bin/env bash
# Simple environment tuning for CPU parallelism to speed up indexing/training.
set -euo pipefail

# number of CPU cores
NPROC=$(nproc 2>/dev/null || echo 1)

: ${OMP_NUM_THREADS:=$NPROC}
: ${MKL_NUM_THREADS:=$NPROC}
: ${OPENBLAS_NUM_THREADS:=$NPROC}
: ${NUMEXPR_NUM_THREADS:=$NPROC}

export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS

echo "[env_setup] CPU cores: $NPROC"
echo "[env_setup] OMP_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$MKL_NUM_THREADS OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
