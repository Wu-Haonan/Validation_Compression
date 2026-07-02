#!/usr/bin/env python3
"""
split_ground_truth_per_accession.py

Splits compute_full_ground_truth's single combined CSV (all pairs) into
per-accession files (col_acc,jaccard_raw), matching the exact format
Step_2's ground_truth_full_row_sampler.cpp produces -- so Step_2's
full_row_compare.py can be reused UNCHANGED here to validate that the
compressed matrix matches ground truth for this small Step_3 dataset.

Accessions with ZERO real neighbors (e.g. zero_neighbor_X, row0_marker,
last_row_marker) get an empty (header-only) file written too, since the
combined CSV lists every (row,col) pair including should_exist=0 rows --
so we know about these accessions even though they have no neighbors.

Usage:
    python3 split_ground_truth_per_accession.py \
        --combined_csv step3_full_ground_truth.csv \
        --out_dir step3_ground_truth/
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    full_df = pd.read_csv(args.combined_csv)

    all_accessions = set(full_df["row_acc"]) | set(full_df["col_acc"])
    real_df = full_df[full_df["should_exist"] == 1]

    accessions_with_neighbors = set()
    for row_acc, group in real_df.groupby("row_acc"):
        out_path = os.path.join(args.out_dir, f"{row_acc}.csv")
        group[["col_acc", "jaccard_raw"]].sort_values(
            "jaccard_raw", ascending=False
        ).to_csv(out_path, index=False)
        accessions_with_neighbors.add(row_acc)

    zero_neighbor_accs = all_accessions - accessions_with_neighbors
    for acc in zero_neighbor_accs:
        out_path = os.path.join(args.out_dir, f"{acc}.csv")
        with open(out_path, "w") as f:
            f.write("col_acc,jaccard_raw\n")  # header only, zero rows

    print(f"Wrote {len(accessions_with_neighbors)} per-accession files with neighbors, "
          f"{len(zero_neighbor_accs)} empty (zero-neighbor) files, to {args.out_dir}")


if __name__ == "__main__":
    main()