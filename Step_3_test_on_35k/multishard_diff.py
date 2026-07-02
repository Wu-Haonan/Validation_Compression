#!/usr/bin/env python3
"""
multishard_diff.py

Multi-shard consistency check: running pairwise_comp_optimized on the SAME
dataset with --num_shards 1 vs 2 vs 3 should produce IDENTICAL results
after merging/querying -- sharding is purely a parallelization strategy,
not something that should change any value. Unlike other Step_1/2/3
comparisons, NO tolerance is appropriate here: these are two runs of the
literal same deterministic computation, just partitioned differently, so
any difference at all (existence or value) is a real bug.

Usage:
    python3 multishard_diff.py \
        --query_accessions step3_query_accessions.txt \
        --decoded_dir_a decoded_shards1 \
        --decoded_dir_b decoded_shards2 \
        --label_a "num_shards=1" \
        --label_b "num_shards=2" \
        --out_mismatch multishard_mismatches_1v2.csv
"""

import argparse
import os
import pandas as pd


def load_full_row(decoded_dir, suffix, acc):
    filename = f"{acc}.csv" if suffix == "" else f"{acc}{suffix}"
    path = os.path.join(decoded_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return dict(zip(df["ID"], df["Jaccard"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_accessions", required=True)
    parser.add_argument("--decoded_dir_a", required=True)
    parser.add_argument("--decoded_dir_b", required=True)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    parser.add_argument("--decoded_suffix", default="",
                         help="suffix INCLUDING .csv, e.g. '_step3.csv'. Empty means {acc}.csv.")
    parser.add_argument("--out_mismatch", default="multishard_mismatches.csv")
    args = parser.parse_args()

    with open(args.query_accessions) as f:
        accs = [line.strip() for line in f if line.strip()]

    mismatches = []
    for acc in accs:
        a = load_full_row(args.decoded_dir_a, args.decoded_suffix, acc)
        b = load_full_row(args.decoded_dir_b, args.decoded_suffix, acc)

        if a is None and b is None:
            # Consistent: both shardings produced no file for this row
            # (e.g. a genuinely zero-neighbor row -- pc_mat::query() never
            # sets self_id for it, so the CLI skips writing a file, on
            # BOTH runs). Not a mismatch.
            continue
        if a is None or b is None:
            mismatches.append((acc, None, f"missing file on only one side: a={a is None}, b={b is None}"))
            continue

        keys_a, keys_b = set(a.keys()), set(b.keys())
        for k in keys_a - keys_b:
            mismatches.append((acc, k, f"present in {args.label_a} but not {args.label_b} (value={a[k]})"))
        for k in keys_b - keys_a:
            mismatches.append((acc, k, f"present in {args.label_b} but not {args.label_a} (value={b[k]})"))
        for k in keys_a & keys_b:
            if a[k] != b[k]:  # NO tolerance -- same deterministic computation, must match exactly
                mismatches.append((acc, k, f"value differs: {args.label_a}={a[k]}, {args.label_b}={b[k]}"))

    print(f"Checked {len(accs)} accessions between {args.label_a} and {args.label_b}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        pd.DataFrame(mismatches, columns=["accession", "neighbor", "issue"]).to_csv(args.out_mismatch, index=False)
        print(f"Saved to {args.out_mismatch}")
        for m in mismatches[:20]:
            print(m)
    else:
        print(f"[OK] {args.label_a} and {args.label_b} are byte-for-byte identical")


if __name__ == "__main__":
    main()