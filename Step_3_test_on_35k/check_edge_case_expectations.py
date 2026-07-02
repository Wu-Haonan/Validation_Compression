#!/usr/bin/env python3
"""
check_edge_case_expectations.py

Checks the STRUCTURAL expectations behind each edge-case group in
generate_edge_case_dataset.py's manifest, against compute_full_ground_truth's
combined CSV and the db_folder's actual vector ordering:

  - single_neighbor_A / _mut : exactly 1 neighbor each (its own pair partner)
  - zero_neighbor_X          : exactly 0 neighbors
  - near_identical_A / _mut  : exactly 1 neighbor, with a high jaccard
                                (informational -- prints the actual value,
                                no hard pass/fail cutoff since the target
                                was 0.97, not exactly 1)
  - boundary_XX / _mut       : reports which side of the real threshold
                                each pair landed on (should_exist True/False)
                                -- WARNS (does not fail) if all boundary
                                pairs landed on the same side, since the
                                whole point of this group was to straddle
                                the threshold for diagnostic purposes
  - background_XX / _mut     : exactly 1 neighbor each (typical case)
  - row0_marker               : must be vector index 0 in vector_norms.txt
  - last_row_marker           : must be the LAST vector index in vector_norms.txt

Usage:
    python3 check_edge_case_expectations.py \
        --manifest step3_manifest.csv \
        --ground_truth step3_full_ground_truth.csv \
        --db_folder /path/to/step3_db
"""

import argparse
import pandas as pd


def neighbor_count(gt_df, acc):
    return int(((gt_df["row_acc"] == acc) & (gt_df["should_exist"] == 1)).sum())


def neighbor_jaccard(gt_df, acc, partner):
    row = gt_df[(gt_df["row_acc"] == acc) & (gt_df["col_acc"] == partner)]
    if row.empty:
        return None
    return float(row.iloc[0]["jaccard_raw"]) if row.iloc[0]["should_exist"] == 1 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ground_truth", required=True)
    parser.add_argument("--db_folder", required=True)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    gt = pd.read_csv(args.ground_truth)

    failures = []
    warnings = []

    def fail(msg):
        failures.append(msg)
        print(f"[FAIL] {msg}")

    def warn(msg):
        warnings.append(msg)
        print(f"[WARN] {msg}")

    def ok(msg):
        print(f"[OK]   {msg}")

    # ---- single_neighbor ----
    for acc in ["single_neighbor_A", "single_neighbor_A_mut"]:
        n = neighbor_count(gt, acc)
        if n == 1:
            ok(f"{acc}: exactly 1 neighbor, as expected")
        else:
            fail(f"{acc}: expected exactly 1 neighbor, found {n}")

    # ---- zero_neighbor ----
    n = neighbor_count(gt, "zero_neighbor_X")
    if n == 0:
        ok("zero_neighbor_X: exactly 0 neighbors, as expected")
    else:
        fail(f"zero_neighbor_X: expected 0 neighbors, found {n}")

    # ---- near_identical ----
    n = neighbor_count(gt, "near_identical_A")
    j = neighbor_jaccard(gt, "near_identical_A", "near_identical_A_mut")
    if n == 1 and j is not None:
        ok(f"near_identical_A: exactly 1 neighbor, jaccard_raw={j:.5f} "
           f"(quantized ~{round(j*255)}/255)")
    else:
        fail(f"near_identical_A: expected exactly 1 neighbor with a valid jaccard, "
             f"found n={n}, jaccard={j}")

    # ---- boundary_threshold group ----
    boundary_rows = manifest[manifest["group"] == "boundary_threshold"]
    boundary_bases = sorted(boundary_rows[boundary_rows["role"] == "base"]["accession"].tolist())
    sides = []
    for base in boundary_bases:
        mut = base + "_mut"
        n = neighbor_count(gt, base)
        j = neighbor_jaccard(gt, base, mut)
        landed = "ABOVE threshold (exists)" if n == 1 else "BELOW threshold (filtered out)"
        sides.append(n == 1)
        print(f"       boundary pair {base}/{mut}: n_neighbors={n}, jaccard_raw={j}, {landed}")
    if boundary_bases:
        if all(sides) or not any(sides):
            warn(f"all {len(boundary_bases)} boundary pairs landed on the SAME side of the threshold -- "
                 f"the group didn't actually straddle the real cutoff; consider widening "
                 f"--boundary_spread or recalibrating and regenerating")
        else:
            ok(f"boundary group straddled the threshold: "
               f"{sum(sides)} above / {len(sides)-sum(sides)} below, out of {len(sides)} pairs")

    # ---- background (typical case) ----
    background_rows = manifest[manifest["group"] == "background"]
    background_bases = sorted(background_rows[background_rows["role"] == "base"]["accession"].tolist())
    bg_fail = 0
    for base in background_bases:
        n = neighbor_count(gt, base)
        if n != 1:
            bg_fail += 1
            fail(f"background pair {base}: expected exactly 1 neighbor, found {n}")
    if bg_fail == 0 and background_bases:
        ok(f"all {len(background_bases)} background pairs have exactly 1 neighbor, as expected")

    # ---- row index checks (row0_marker / last_row_marker) ----
    with open(args.db_folder.rstrip("/") + "/vector_norms.txt") as f:
        accession_order = [line.split()[0] for line in f if line.strip()]

    if accession_order and accession_order[0] == "row0_marker":
        ok("row0_marker is at vector index 0, as expected")
    else:
        fail(f"row0_marker expected at index 0, but index 0 is "
             f"'{accession_order[0] if accession_order else '<empty>'}'")

    if accession_order and accession_order[-1] == "last_row_marker":
        ok(f"last_row_marker is at the last vector index ({len(accession_order)-1}), as expected")
    else:
        fail(f"last_row_marker expected at the last index, but last index is "
             f"'{accession_order[-1] if accession_order else '<empty>'}'")

    print(f"\n===== Summary: {len(failures)} failures, {len(warnings)} warnings =====")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()