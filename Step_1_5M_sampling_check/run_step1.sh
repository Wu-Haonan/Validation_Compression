#!/bin/bash
# run_step1.sh
#
# Step 1: random sampling validation against the 5M production dataset,
# PLUS an "extreme" mode targeting vectors at the norm extremes.
#
# Sampling design (random mode): sample NUM_ROWS distinct rows and
# NUM_COLS distinct cols (disjoint sets), take their full cross product as
# the test set (NUM_ROWS * NUM_COLS pairs, no self-pairs, no wasted
# queries). A single sliced matrix query (--row_file/--col_file) returns
# exactly this slice in one call.
#
# Sampling design (extreme mode): take the N accessions with the SMALLEST
# vector norm and the N accessions with the LARGEST vector norm (2N total),
# used as BOTH rows and cols, producing a full (2N x 2N) submatrix. This
# targets vectors at the extremes of length/complexity, which random
# sampling is very unlikely to specifically hit.
#
# Both the C++ query_pc_mat tool and the Python read_pc_mat.py script are
# run, and EACH is independently compared against the same ground truth
# (NOT a C++-vs-Python cross-check -- two separate ground-truth
# validations; if BOTH disagree, that points to the shared compression/
# encoding logic rather than one specific tool's read path).
#
# QUANTIZATION: ground_truth_sampler.cpp computes BOTH raw and quantized
# jaccard in C++ (using std::round, matching production exactly) --
# compare_results.py reads jaccard_quantized directly, it does not
# recompute it in Python (avoids Python/C++ rounding discrepancies at
# exact .5 boundaries).
#
# Usage:
#   bash run_step1.sh
#
# Please double-check the path variables below before running.

set -e

# ===== Config, edit as needed =====
DB_FOLDER="/scratch/mgs_project/db"
MATRIX_FOLDER="/scratch/mgs_project/matrix_unzipped"
QUERY_PC_MAT_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/query_pc_mat"
READ_PC_MAT_PY="/scratch/hvw5426/metagenome_vector_sketches/src/read_pc_mat.py"
EIGEN_INCLUDE="/scratch/hvw5426/metagenome_vector_sketches/include/Eigen"
NUM_ROWS=100
NUM_COLS=100
EXTREME_N=50          # extreme mode: N smallest-norm + N largest-norm accessions
SEED=42
OUTPUT_PREFIX="sample1"
EXTREME_PREFIX="sample1_extreme"
THREADS=8
# ===================================

echo "===== Step 0: Compile ground_truth_sampler ====="
g++ -std=c++17 -O3 -ffast-math -I"$EIGEN_INCLUDE" -o ground_truth_sampler ground_truth_sampler.cpp
# NOTE: -march=native was REMOVED -- it generates code using the CPU
# instruction set of the machine you COMPILE on, which can crash
# (typically SIGILL / "Aborted (core dumped)") if you then RUN the binary
# on a different node with an older/different CPU (common on shared
# clusters where login nodes and compute nodes differ). If you know for
# certain compilation and execution always happen on the identical CPU
# model, -march=native can be added back for a modest speed gain.

# ---------------------------------------------------------------------
# RANDOM MODE
# ---------------------------------------------------------------------
echo "===== Step 1 [random]: Generate ground truth (row x col cross product) ====="
./ground_truth_sampler random "$DB_FOLDER" "$NUM_ROWS" "$NUM_COLS" "$OUTPUT_PREFIX" "$SEED"

echo "===== Step 2a [random]: Sliced query via C++ query_pc_mat ====="
"$QUERY_PC_MAT_BIN" --matrix "$MATRIX_FOLDER" \
    --db "$DB_FOLDER" \
    --row_file "${OUTPUT_PREFIX}_row_file.txt" \
    --col_file "${OUTPUT_PREFIX}_col_file.txt" \
    --write_to_file "${OUTPUT_PREFIX}_slice_cpp.csv" --thread "$THREADS"

echo "===== Step 2b [random]: Sliced query via Python read_pc_mat.py (stdout only) ====="
python3 "$READ_PC_MAT_PY" --matrix "$MATRIX_FOLDER" \
    --db "$DB_FOLDER" \
    --row_file "${OUTPUT_PREFIX}_row_file.txt" \
    --col_file "${OUTPUT_PREFIX}_col_file.txt" \
    > "${OUTPUT_PREFIX}_slice_py_raw.txt"

echo "===== Step 2c [random]: Convert read_pc_mat.py's stdout into a real CSV ====="
python3 convert_py_stdout_to_csv.py \
    --input "${OUTPUT_PREFIX}_slice_py_raw.txt" \
    --output "${OUTPUT_PREFIX}_slice_py.csv"

echo "===== Step 3a [random, independent check #1]: C++ result vs ground truth ====="
python3 compare_results.py \
    --gt "${OUTPUT_PREFIX}_ground_truth.csv" \
    --slice "${OUTPUT_PREFIX}_slice_cpp.csv" \
    --out_mismatch "mismatches_cpp.csv"

echo "===== Step 3b [random, independent check #2]: Python result vs ground truth ====="
python3 compare_results.py \
    --gt "${OUTPUT_PREFIX}_ground_truth.csv" \
    --slice "${OUTPUT_PREFIX}_slice_py.csv" \
    --out_mismatch "mismatches_py.csv"

# ---------------------------------------------------------------------
# EXTREME MODE (smallest N + largest N vectors by norm, full 2N x 2N)
# ---------------------------------------------------------------------
echo "===== Step 1 [extreme]: Generate ground truth (2N x 2N submatrix of norm extremes) ====="
./ground_truth_sampler extreme "$DB_FOLDER" "$EXTREME_N" "$EXTREME_PREFIX"

echo "===== Step 2a [extreme]: Sliced query via C++ query_pc_mat ====="
"$QUERY_PC_MAT_BIN" --matrix "$MATRIX_FOLDER" \
    --db "$DB_FOLDER" \
    --row_file "${EXTREME_PREFIX}_row_file.txt" \
    --col_file "${EXTREME_PREFIX}_col_file.txt" \
    --write_to_file "${EXTREME_PREFIX}_slice_cpp.csv" --thread "$THREADS"

echo "===== Step 2b [extreme]: Sliced query via Python read_pc_mat.py ====="
python3 "$READ_PC_MAT_PY" --matrix "$MATRIX_FOLDER" \
    --db "$DB_FOLDER" \
    --row_file "${EXTREME_PREFIX}_row_file.txt" \
    --col_file "${EXTREME_PREFIX}_col_file.txt" \
    > "${EXTREME_PREFIX}_slice_py_raw.txt"

echo "===== Step 2c [extreme]: Convert stdout into CSV ====="
python3 convert_py_stdout_to_csv.py \
    --input "${EXTREME_PREFIX}_slice_py_raw.txt" \
    --output "${EXTREME_PREFIX}_slice_py.csv"

echo "===== Step 3a [extreme, independent check #3]: C++ result vs ground truth ====="
python3 compare_results.py \
    --gt "${EXTREME_PREFIX}_ground_truth.csv" \
    --slice "${EXTREME_PREFIX}_slice_cpp.csv" \
    --out_mismatch "mismatches_extreme_cpp.csv"

echo "===== Step 3b [extreme, independent check #4]: Python result vs ground truth ====="
python3 compare_results.py \
    --gt "${EXTREME_PREFIX}_ground_truth.csv" \
    --slice "${EXTREME_PREFIX}_slice_py.csv" \
    --out_mismatch "mismatches_extreme_py.csv"

echo "===== All done -- check mismatches_{cpp,py}.csv and mismatches_extreme_{cpp,py}.csv ====="