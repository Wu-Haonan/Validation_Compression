#!/usr/bin/env python3
"""
find_edge_case_accessions.py

Scans the REAL 35k-subset decoded matrix (--show_all output, one file per
accession that HAS at least one neighbor) plus vector_norms.txt's row
order, to find naturally-occurring edge cases -- no synthetic data needed:

  - zero_neighbor: accessions with NO decoded file at all (pc_mat::query()
    never sets self_id for a genuinely zero-neighbor row, so the CLI never
    writes a file for it -- see Step_2's findings)
  - single_neighbor: accessions whose decoded file has exactly 1 row
  - row0: the first accession in vector_norms.txt's order
  - last_row: the last accession in vector_norms.txt's order

Usage:
    python3 find_edge_case_accessions.py \
        --db_folder step3_db \
        --decoded_dir decoded_shard1 \
        --decoded_suffix _step3.csv \
        --all_accessions step3_all_accessions.txt \
        --out_targets step3_edge_case_targets.txt \
        --out_report step3_edge_case_report.csv \
        --max_per_category 20
"""

import argparse
import os
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_folder", required=True)
    parser.add_argument("--decoded_dir", required=True,
                         help="folder of per-accession --show_all decoded files")
    parser.add_argument("--decoded_suffix", default="",
                         help="suffix INCLUDING .csv, e.g. '_step3.csv'. Empty means {acc}.csv.")
    parser.add_argument("--all_accessions", required=True,
                         help="file listing every accession in this subset (one per line)")
    parser.add_argument("--out_targets", required=True,
                         help="output: accessions to feed into ground_truth_for_specific_rows")
    parser.add_argument("--out_report", required=True,
                         help="output: a csv with one row per found edge case, with its category")
    parser.add_argument("--max_per_category", type=int, default=20,
                         help="cap how many examples of each category to keep "
                              "(zero/single neighbor categories could be large in a dense dataset)")
    args = parser.parse_args()

    with open(args.all_accessions) as f:
        all_accs = [line.strip() for line in f if line.strip()]

    zero_neighbor = []
    single_neighbor = []

    for acc in all_accs:
        filename = f"{acc}.csv" if args.decoded_suffix == "" else f"{acc}{args.decoded_suffix}"
        path = os.path.join(args.decoded_dir, filename)
        if not os.path.exists(path):
            zero_neighbor.append(acc)
            continue
        df = pd.read_csv(path)
        df = df[df["ID"] != acc]  # drop self-pair
        if len(df) == 0:
            zero_neighbor.append(acc)
        elif len(df) == 1:
            single_neighbor.append(acc)

    zero_neighbor = zero_neighbor[:args.max_per_category]
    single_neighbor = single_neighbor[:args.max_per_category]

    # row0 / last_row from vector_norms.txt's actual order
    with open(os.path.join(args.db_folder, "vector_norms.txt")) as f:
        order = [line.split()[0] for line in f if line.strip()]
    row0_acc = order[0] if order else None
    last_row_acc = order[-1] if order else None

    report_rows = []
    for acc in zero_neighbor:
        report_rows.append((acc, "zero_neighbor"))
    for acc in single_neighbor:
        report_rows.append((acc, "single_neighbor"))
    if row0_acc:
        report_rows.append((row0_acc, "row0"))
    if last_row_acc:
        report_rows.append((last_row_acc, "last_row"))

    pd.DataFrame(report_rows, columns=["accession", "category"]).to_csv(args.out_report, index=False)

    target_accs = sorted(set(a for a, _ in report_rows))
    with open(args.out_targets, "w") as f:
        for a in target_accs:
            f.write(a + "\n")

    print(f"Found {len(zero_neighbor)} zero-neighbor accessions (capped at {args.max_per_category})")
    print(f"Found {len(single_neighbor)} single-neighbor accessions (capped at {args.max_per_category})")
    print(f"row0: {row0_acc}")
    print(f"last_row: {last_row_acc}")
    print(f"Wrote {len(target_accs)} unique target accessions to {args.out_targets}")
    print(f"Wrote category report to {args.out_report}")


if __name__ == "__main__":
    main()