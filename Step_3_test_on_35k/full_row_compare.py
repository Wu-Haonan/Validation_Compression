#!/usr/bin/env python3
"""
full_row_compare.py

Check #1/#2: for each sampled accession, compare its FULL (unquantized)
ground-truth neighbor list (from ground_truth_full_row_sampler.cpp) against
the decoded full neighbor list (from query_pc_mat --show_all, or from
direct_query_full.py for the Python module). This validates that the
compressed matrix contains exactly the right set of neighbors with the
right values for an entire row -- not just specific row x col cells like
Step_1.

Self-pairs (accession vs itself, jaccard=1) are intentionally excluded from
the ground truth (see ground_truth_full_row_sampler.cpp) and are dropped
from the decoded list here too before comparing -- self-pairs are already
covered separately by Step_1's self_pair_check.py.

Usage:
    python3 full_row_compare.py \
        --query_accessions sample2_query_accessions.txt \
        --gt_dir sample2_ground_truth \
        --decoded_dir decoded_cpp \
        --decoded_suffix _sample2.csv \
        --out_mismatch full_row_mismatches_cpp.csv
"""

import argparse
import os
import pandas as pd

QUANT_TOL = 1.0 / 255.0


def load_gt(gt_dir, acc):
    path = os.path.join(gt_dir, f"{acc}.csv")
    df = pd.read_csv(path)
    return dict(zip(df["col_acc"], df["jaccard_raw"]))


def load_decoded(decoded_dir, suffix, acc):
    filename = f"{acc}.csv" if suffix == "" else f"{acc}{suffix}"
    path = os.path.join(decoded_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    d = dict(zip(df["ID"], df["Jaccard"]))
    d.pop(acc, None)  # drop self-pair, not modeled in ground truth
    return d


def compare_one(acc, gt_dict, decoded_dict, tol, mismatches):
    if decoded_dict is None:
        if len(gt_dict) == 0:
            # EXPECTED: pc_mat::query() leaves res.self_id empty for
            # zero-neighbor rows (it never gets assigned -- see
            # load_neighbors_for_rows_jaccard_wo_sort's empty-result path),
            # and the calling CLI explicitly skips writing a file when
            # self_id is empty ("Skipping file write"). So a missing file
            # for a row that genuinely has 0 ground-truth neighbors is
            # CORRECT behavior, not a bug.
            return
        mismatches.append((acc, None, "decoded file missing (but ground truth has neighbors -- real miss)"))
        return

    gt_keys = set(gt_dict.keys())
    dec_keys = set(decoded_dict.keys())

    missing = gt_keys - dec_keys
    extra = dec_keys - gt_keys

    for k in missing:
        mismatches.append((acc, k, "in ground truth but missing from decoded"))
    for k in extra:
        mismatches.append((acc, k, f"in decoded but not in ground truth, value={decoded_dict[k]}"))

    for k in gt_keys & dec_keys:
        diff = abs(gt_dict[k] - decoded_dict[k])
        if diff > tol:
            mismatches.append((acc, k, f"value mismatch: raw={gt_dict[k]}, decoded={decoded_dict[k]}, diff={diff:.6f}"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_accessions", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--decoded_dir", required=True)
    parser.add_argument("--decoded_suffix", default="",
                         help="suffix INCLUDING .csv extension appended to accession for decoded "
                              "filenames, e.g. '_sample2.csv' if files are named {acc}_sample2.csv. "
                              "Leave empty (default) if files are named {acc}.csv directly.")
    parser.add_argument("--tol", type=float, default=QUANT_TOL)
    parser.add_argument("--out_mismatch", default="full_row_mismatches.csv")
    args = parser.parse_args()

    with open(args.query_accessions) as f:
        accs = [line.strip() for line in f if line.strip()]

    mismatches = []
    total_gt_entries = 0

    for acc in accs:
        gt_dict = load_gt(args.gt_dir, acc)
        total_gt_entries += len(gt_dict)
        decoded_dict = load_decoded(args.decoded_dir, args.decoded_suffix, acc)
        compare_one(acc, gt_dict, decoded_dict, args.tol, mismatches)

    print(f"Checked {len(accs)} accessions, {total_gt_entries} total ground-truth neighbor entries")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        pd.DataFrame(mismatches, columns=["accession", "neighbor", "issue"]).to_csv(args.out_mismatch, index=False)
        print(f"Saved to {args.out_mismatch}")
        print("\nFirst 20 mismatches:")
        for m in mismatches[:20]:
            print(m)
    else:
        print("[OK] full-row check passed for all sampled accessions")


if __name__ == "__main__":
    main()