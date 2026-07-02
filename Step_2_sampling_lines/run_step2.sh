#!/bin/bash
# run_step2.sh
#
# Step 2: full-row validation + Top-K validation, treating production's
# sort/tie-break logic as a black box. PLUS an "extreme" mode targeting
# the smallest-norm and largest-norm accessions as query rows.
#
# Sampling design (random mode): pick N random distinct accessions, stream
# through the ENTIRE vectors.bin once (Eigen GEMM) to compute each
# accession's FULL, unquantized neighbor list as ground truth.
#
# Sampling design (extreme mode): pick the N smallest-norm + N largest-norm
# accessions (2N total) as query rows, each queried against the ENTIRE
# database (not just against each other) -- targets whether extreme-norm
# vectors behave correctly as query rows in realistic usage.
#
# Then independently check, for BOTH the C++ query_pc_mat tool and the
# Python module (via direct_query_full.py, bypassing read_pc_mat.py's
# hardcoded top-10 print limit):
#   1. full_row_compare.py     -- does the ENTIRE decoded neighbor list
#      (--show_all) for each row match ground truth (existence + value,
#      tol = 1/255)?
#   2. topk_compare.py         -- does the Top-K SET, derived by taking
#      --show_all's full list and re-sorting it OURSELVES, match ground
#      truth's Top-K set, for K = 10/20/50/100/1000/10000?
#   3. topk_function_check.py -- does the --top K PARAMETER ITSELF (no
#      --show_all, no re-sorting on our end) return the correct Top-K set?
#      (C++ query_pc_mat only.)
#
# QUANTIZATION: ground_truth_full_row_sampler.cpp computes BOTH raw and
# quantized jaccard in C++ (std::round, matching production exactly) --
# downstream Python scripts read jaccard_quantized directly, never
# recompute it.
#
# As in Step_1, these are all INDEPENDENT checks against the same ground
# truth, not a C++-vs-Python cross-check.
#
# Usage:
#   bash run_step2.sh
#
# Please double-check the path variables below before running.

set -e

# ===== Config, edit as needed =====
DB_FOLDER="/scratch/mgs_project/db"
MATRIX_FOLDER="/scratch/mgs_project/matrix_unzipped"
QUERY_PC_MAT_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/query_pc_mat"
READ_PC_MAT_MODULE_DIR="/scratch/hvw5426/metagenome_vector_sketches/src"
EIGEN_INCLUDE="/scratch/hvw5426/metagenome_vector_sketches/include/Eigen"
N=50
EXTREME_N=25           # extreme mode: N smallest-norm + N largest-norm accessions
SEED=42
OUTPUT_PREFIX="sample2"
EXTREME_PREFIX="sample2_extreme"
THREADS=8
KS="10 20 50 100 1000 10000"
# ===================================

echo "===== Step 0: Compile ground_truth_full_row_sampler (Eigen-accelerated) ====="
g++ -std=c++17 -O3 -march=native -ffast-math -I"$EIGEN_INCLUDE" -o ground_truth_full_row_sampler ground_truth_full_row_sampler.cpp

run_full_pipeline() {
    local PREFIX=$1
    local DECODED_CPP="decoded_cpp_${PREFIX}"
    local DECODED_PY="decoded_py_${PREFIX}"
    local DECODED_TOP_BASE="decoded_top_cpp_${PREFIX}"

    echo "===== [$PREFIX] Step 2a: C++ query_pc_mat --show_all (decoded full row per accession) ====="
    mkdir -p "$DECODED_CPP"
    "$QUERY_PC_MAT_BIN" --matrix "$MATRIX_FOLDER" --db "$DB_FOLDER" \
        --query_file "${PREFIX}_query_accessions.txt" --show_all \
        --write_to_file "${DECODED_CPP}/${PREFIX}.csv" --thread "$THREADS"

    echo "===== [$PREFIX] Step 2b: Python direct module query (full neighbor list) ====="
    mkdir -p "$DECODED_PY"
    python3 direct_query_full.py --matrix "$MATRIX_FOLDER" --db "$DB_FOLDER" \
        --query_file "${PREFIX}_query_accessions.txt" \
        --module_dir "$READ_PC_MAT_MODULE_DIR" \
        --out_dir "$DECODED_PY"

    echo "===== [$PREFIX] Step 3a [check #1]: Full-row comparison, C++ vs ground truth ====="
    python3 full_row_compare.py \
        --query_accessions "${PREFIX}_query_accessions.txt" \
        --gt_dir "${PREFIX}_ground_truth" \
        --decoded_dir "$DECODED_CPP" \
        --decoded_suffix "_${PREFIX}.csv" \
        --out_mismatch "full_row_mismatches_cpp_${PREFIX}.csv"

    echo "===== [$PREFIX] Step 3b [check #2]: Full-row comparison, Python vs ground truth ====="
    python3 full_row_compare.py \
        --query_accessions "${PREFIX}_query_accessions.txt" \
        --gt_dir "${PREFIX}_ground_truth" \
        --decoded_dir "$DECODED_PY" \
        --out_mismatch "full_row_mismatches_py_${PREFIX}.csv"

    echo "===== [$PREFIX] Step 4a [check #3]: Top-K SET comparison, C++ vs ground truth ====="
    python3 topk_compare.py \
        --query_accessions "${PREFIX}_query_accessions.txt" \
        --gt_dir "${PREFIX}_ground_truth" \
        --decoded_dir "$DECODED_CPP" \
        --decoded_suffix "_${PREFIX}.csv" \
        --ks $KS \
        --out_mismatch "topk_mismatches_cpp_${PREFIX}.csv"

    echo "===== [$PREFIX] Step 4b [check #4]: Top-K SET comparison, Python vs ground truth ====="
    python3 topk_compare.py \
        --query_accessions "${PREFIX}_query_accessions.txt" \
        --gt_dir "${PREFIX}_ground_truth" \
        --decoded_dir "$DECODED_PY" \
        --ks $KS \
        --out_mismatch "topk_mismatches_py_${PREFIX}.csv"

    echo "===== [$PREFIX] Step 5 [check #5]: --top K FEATURE ITSELF (C++ only) ====="
    mkdir -p "$DECODED_TOP_BASE"
    for K in $KS; do
        K_PLUS_1=$((K + 1))   # +1 for self-pair, see Step_1 findings
        OUT_DIR="${DECODED_TOP_BASE}/top_${K}"
        mkdir -p "$OUT_DIR"
        "$QUERY_PC_MAT_BIN" --matrix "$MATRIX_FOLDER" --db "$DB_FOLDER" \
            --query_file "${PREFIX}_query_accessions.txt" \
            --top "$K_PLUS_1" \
            --write_to_file "${OUT_DIR}/${PREFIX}.csv" --thread "$THREADS"
    done
    python3 topk_function_check.py \
        --query_accessions "${PREFIX}_query_accessions.txt" \
        --gt_dir "${PREFIX}_ground_truth" \
        --decoded_base_dir "$DECODED_TOP_BASE" \
        --decoded_suffix "_${PREFIX}.csv" \
        --ks $KS \
        --out_mismatch "topk_function_mismatches_cpp_${PREFIX}.csv"
}

# ---------------------------------------------------------------------
# RANDOM MODE
# ---------------------------------------------------------------------
echo "===== Step 1 [random]: Generate ground truth for $N randomly-sampled accessions ====="
./ground_truth_full_row_sampler random "$DB_FOLDER" "$N" "$OUTPUT_PREFIX" "$SEED"
run_full_pipeline "$OUTPUT_PREFIX"

# ---------------------------------------------------------------------
# EXTREME MODE (smallest N + largest N accessions by norm)
# ---------------------------------------------------------------------
echo "===== Step 1 [extreme]: Generate ground truth for $EXTREME_N smallest + $EXTREME_N largest-norm accessions ====="
./ground_truth_full_row_sampler extreme "$DB_FOLDER" "$EXTREME_N" "$EXTREME_PREFIX"
run_full_pipeline "$EXTREME_PREFIX"

echo "===== All done -- check full_row_mismatches_*.csv, topk_mismatches_*.csv, topk_function_mismatches_*.csv ====="
echo "      (both for prefix '${OUTPUT_PREFIX}' [random] and '${EXTREME_PREFIX}' [extreme])"