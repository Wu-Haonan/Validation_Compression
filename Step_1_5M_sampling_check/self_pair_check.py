#!/usr/bin/env python3
"""
self_pair_check.py

Supplementary check: for every sampled row, verify that querying itself
(self-pair) returns Jaccard = 1.0. We already confirmed on real production
data that self-pairs are kept in the results, e.g.:
  querying DRR005282 -> the first neighbor is DRR005282 itself, Jaccard=1

Usage:
    python3 self_pair_check.py \
        --rows sample1_unique_rows.txt \
        --decoded_dir decoded_outputs \
        --decoded_suffix _sample1.csv
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True, help="unique_rows.txt, one accession per line")
    parser.add_argument("--decoded_dir", required=True)
    parser.add_argument("--decoded_suffix", required=True)
    parser.add_argument("--float_tol", type=float, default=1e-6)
    args = parser.parse_args()

    with open(args.rows) as f:
        accessions = [line.strip() for line in f if line.strip()]

    fail = []
    missing_file = []

    for acc in accessions:
        path = os.path.join(args.decoded_dir, f"{acc}{args.decoded_suffix}")
        if not os.path.exists(path):
            missing_file.append(acc)
            continue
        df = pd.read_csv(path)
        neighbors = dict(zip(df["ID"], df["Jaccard"]))
        self_jaccard = neighbors.get(acc)
        if self_jaccard is None:
            fail.append((acc, "self not found in its own neighbor list"))
        elif abs(self_jaccard - 1.0) > args.float_tol:
            fail.append((acc, f"self jaccard != 1.0, actual value={self_jaccard}"))

    print(f"Checked {len(accessions)} accessions")
    print(f"Missing files (row has zero neighbors or query failed): {len(missing_file)}")
    print(f"Self-pair anomalies: {len(fail)}")

    if fail:
        print("\nAnomaly details (first 20):")
        for f_ in fail[:20]:
            print(f_)
        pd.DataFrame(fail, columns=["accession", "issue"]).to_csv("self_pair_mismatches.csv", index=False)
    else:
        print("\n[OK] All sampled rows correctly return Jaccard=1.0 when querying themselves")


if __name__ == "__main__":
    main()