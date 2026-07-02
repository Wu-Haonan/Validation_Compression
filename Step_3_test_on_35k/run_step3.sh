#!/bin/bash
# run_step3.sh
#
# Step 3 (revised): uses the REAL 35k-accession subset (already-sampled
# from the production 5M dataset, with real sourmash .sig.zip sketches),
# instead of synthetic data. No vectorization-tool mismatch concerns since
# this is genuine production-style data.
#
# Two things this checks that Step_1/Step_2 (on the full 5M dataset)
# genuinely CANNOT check:
#   1. Multi-shard consistency: --num_shards 1 vs --num_shards N on the
#      SAME data must produce byte-for-byte identical decoded results.
#      The production 5M matrix was already built with a fixed shard
#      count, so there's no way to re-test this on it directly.
#   2. Naturally-occurring edge-case rows (zero-neighbor, single-neighbor,
#      row index 0, last row index) -- Step_1/2's random sampling has only
#      a ~1/35000 chance of hitting row 0 or the last row specifically,
#      and zero/single-neighbor rows may or may not have been hit by
#      chance. Here we exhaustively SCAN for them instead of hoping
#      random sampling found them.
#
# Usage:
#   bash run_step3.sh

set -e

# ===== Config, edit as needed =====
SIGS_FOLDER="/scratch/Sample_Logan_sketches/sigs_dna"
DB_FOLDER="step3_db/"   # NOTE: trailing slash is required -- pairwise_comp_optimized's
                        # main() has a bug where it checks db_folder's trailing slash
                        # but appends '/' to output_folder instead, so db_folder NEVER
                        # gets auto-corrected. Passing it pre-slashed works around this.
PROJECT_EVERYTHING_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/project_everything"
PAIRWISE_COMP_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/pairwise_comp_optimized"
QUERY_PC_MAT_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/query_pc_mat"
EIGEN_INCLUDE="/scratch/hvw5426/metagenome_vector_sketches/include/Eigen"
DIMENSION=2048
THREADS=8
NUM_SHARDS_B=5   # the "other" shard count to test against num_shards=1
MAX_PER_CATEGORY=20
# ===================================

echo "===== Step 0: Build vectors.bin from the real sig.zip subset (project_everything) ====="
if [ ! -f "${DB_FOLDER}/vectors.bin" ]; then
    mkdir -p "$DB_FOLDER"
    HASH_FILE="step3_hashes.bin"
    "$PROJECT_EVERYTHING_BIN" convert "$SIGS_FOLDER" "$HASH_FILE" -t "$THREADS"
    "$PROJECT_EVERYTHING_BIN" sketch "$HASH_FILE" "$DB_FOLDER" -t "$THREADS" -d "$DIMENSION"
else
    echo "(${DB_FOLDER}/vectors.bin already exists, skipping)"
fi

# List of every accession in this subset, in vector_norms.txt's order
awk '{print $1}' "${DB_FOLDER}/vector_norms.txt" > step3_all_accessions.txt
echo "Subset size: $(wc -l < step3_all_accessions.txt) accessions"

echo "===== Step 1: Compile ground_truth_for_specific_rows ====="
g++ -std=c++17 -O2 -I"$EIGEN_INCLUDE" -o ground_truth_for_specific_rows ground_truth_for_specific_rows.cpp

echo "===== Step 2: Build compressed matrix A (--num_shards 1) ====="
OUTPUT_A="matrix_shard1"
mkdir -p "$OUTPUT_A"
"$PAIRWISE_COMP_BIN" --db "$DB_FOLDER" --max_memory_gb 8 --num_threads "$THREADS" \
    --output_folder "$OUTPUT_A" --num_shards 1 --shard_idx 0

echo "===== Step 3: Build compressed matrix B (--num_shards $NUM_SHARDS_B, SAME data) ====="
OUTPUT_B="matrix_shard${NUM_SHARDS_B}"
mkdir -p "$OUTPUT_B"
for ((SHARD_IDX=0; SHARD_IDX<NUM_SHARDS_B; SHARD_IDX++)); do
    "$PAIRWISE_COMP_BIN" --db "$DB_FOLDER" --max_memory_gb 8 --num_threads "$THREADS" \
        --output_folder "$OUTPUT_B" --num_shards "$NUM_SHARDS_B" --shard_idx "$SHARD_IDX"
done

echo "===== Step 4: Decode BOTH matrices (--show_all) for ALL accessions in the subset ====="
DECODED_A="decoded_shard1"
DECODED_B="decoded_shard${NUM_SHARDS_B}"
mkdir -p "$DECODED_A" "$DECODED_B"

"$QUERY_PC_MAT_BIN" --matrix "$OUTPUT_A" --db "$DB_FOLDER" \
    --query_file step3_all_accessions.txt --show_all \
    --write_to_file "${DECODED_A}/step3.csv" --thread "$THREADS"

"$QUERY_PC_MAT_BIN" --matrix "$OUTPUT_B" --db "$DB_FOLDER" \
    --query_file step3_all_accessions.txt --show_all \
    --write_to_file "${DECODED_B}/step3.csv" --thread "$THREADS"

echo "===== Step 5: [check #1] Multi-shard consistency (num_shards=1 vs num_shards=$NUM_SHARDS_B) ====="
python3 multishard_diff.py \
    --query_accessions step3_all_accessions.txt \
    --decoded_dir_a "$DECODED_A" \
    --decoded_dir_b "$DECODED_B" \
    --decoded_suffix "_step3.csv" \
    --label_a "num_shards=1" \
    --label_b "num_shards=${NUM_SHARDS_B}" \
    --out_mismatch multishard_mismatches_1v${NUM_SHARDS_B}.csv

echo "===== Step 6: Scan for naturally-occurring edge cases (zero/single-neighbor, row0, last_row) ====="
python3 find_edge_case_accessions.py \
    --db_folder "$DB_FOLDER" \
    --decoded_dir "$DECODED_A" \
    --decoded_suffix "_step3.csv" \
    --all_accessions step3_all_accessions.txt \
    --out_targets step3_edge_case_targets.txt \
    --out_report step3_edge_case_report.csv \
    --max_per_category "$MAX_PER_CATEGORY"

echo "===== Step 7: Compute independent ground truth for those specific edge-case accessions ====="
./ground_truth_for_specific_rows "$DB_FOLDER" step3_edge_case_targets.txt step3_edge_cases

echo "===== Step 8: [check #2] Verify edge-case accessions against ground truth (full-row compare) ====="
python3 full_row_compare.py \
    --query_accessions step3_edge_case_targets.txt \
    --gt_dir step3_edge_cases_ground_truth \
    --decoded_dir "$DECODED_A" \
    --decoded_suffix "_step3.csv" \
    --out_mismatch full_row_mismatches_edge_cases.csv

echo "===== All done -- check multishard_mismatches_1v${NUM_SHARDS_B}.csv, "
echo "      step3_edge_case_report.csv (what was found), and full_row_mismatches_edge_cases.csv ====="