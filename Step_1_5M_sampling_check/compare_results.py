#!/usr/bin/env python3
"""
compare_results.py

Compares ground truth (from ground_truth_sampler.cpp, which now computes
BOTH the raw jaccard AND its quantized integer value, using C++'s
std::round -- the exact same rounding production code uses) against a
single sliced matrix query result (--row_file/--col_file).

IMPORTANT: this script does NOT re-quantize anything itself. It reads the
jaccard_quantized column directly from the ground truth CSV (already
computed in C++) and compares the decoded value's quantized form against
it. This avoids any Python-vs-C++ rounding discrepancy at exact .5
boundaries (Python's round() uses banker's rounding; C++'s std::round()
rounds half away from zero) -- see project history for why this matters.

Usage:
    python3 compare_results.py \
        --gt sample1_ground_truth.csv \
        --slice test_slice.csv
"""

import argparse
import pandas as pd

QUANT_LEVELS = 255


def main():
    parser = argparse.ArgumentParser(description="Compare ground truth vs a sliced matrix query result")
    parser.add_argument("--gt", required=True, help="ground_truth csv path (must include jaccard_quantized column)")
    parser.add_argument("--slice", required=True,
                         help="sliced matrix csv (pivoted: rows=row_acc, columns=col_acc), "
                              "from either query_pc_mat or read_pc_mat.py")
    parser.add_argument("--out_mismatch", default="mismatches.csv", help="output csv for mismatches")
    args = parser.parse_args()

    gt = pd.read_csv(args.gt)
    if "jaccard_quantized" not in gt.columns:
        raise ValueError("ground truth CSV is missing 'jaccard_quantized' -- regenerate it with the "
                          "updated ground_truth_sampler.cpp (quantization now happens in C++).")
    print(f"Total ground truth pairs: {len(gt)}")

    matrix = pd.read_csv(args.slice, index_col=0)
    print(f"Slice matrix shape: {matrix.shape}")

    # NOTE: query_pc_mat's output has a trailing comma per row, which makes
    # pandas parse an extra all-NaN "Unnamed: N" column. The "Accession"
    # header label itself is harmless. Drop any all-NaN columns to clean
    # this up regardless of which tool produced the file.
    matrix = pd.read_csv(args.slice, index_col=0)
    if matrix.index.name == "Accession":
        matrix.index.name = None
    matrix = matrix.dropna(axis=1, how="all")

    results = {
        "should_exist_but_missing": 0,
        "should_not_exist_but_found": 0,
        "value_mismatch": 0,
        "match": 0,
        "both_absent_ok": 0,
        "row_or_col_not_in_slice": 0,
    }
    mismatch_details = []

    for _, row in gt.iterrows():
        row_acc, col_acc = row["row_acc"], row["col_acc"]
        should_exist = bool(row["should_exist"])
        q_expected = row["jaccard_quantized"]  # already computed in C++ via std::round

        if row_acc not in matrix.index or col_acc not in matrix.columns:
            results["row_or_col_not_in_slice"] += 1
            mismatch_details.append((row_acc, col_acc, "row or col missing from the slice matrix"))
            continue

        cell_value = matrix.loc[row_acc, col_acc]
        found = cell_value != 0.0

        if should_exist and not found:
            results["should_exist_but_missing"] += 1
            mismatch_details.append((row_acc, col_acc, "ground truth says it should exist, but slice cell is 0"))
        elif not should_exist and found:
            results["should_not_exist_but_found"] += 1
            mismatch_details.append((row_acc, col_acc, f"ground truth says it should be filtered by threshold, but slice cell is nonzero: {cell_value}"))
        elif should_exist and found:
            # Compare the DECODED value's implied quantized integer against
            # the C++-computed q_expected. The decoded cell_value is itself
            # quantized_int/255 (dequantized), so round-tripping it back
            # should reproduce the exact integer that was actually stored.
            q_decoded = round(cell_value * QUANT_LEVELS)
            if q_decoded != q_expected:
                results["value_mismatch"] += 1
                mismatch_details.append((row_acc, col_acc,
                    f"quantized value mismatch: expected_q={q_expected}, decoded_q={q_decoded} "
                    f"(expected_raw_q/255={q_expected/QUANT_LEVELS}, decoded_cell={cell_value})"))
            else:
                results["match"] += 1
        else:
            results["both_absent_ok"] += 1

    print("\n===== Validation Results =====")
    for k, v in results.items():
        print(f"{k}: {v}")

    total_problems = (results["should_exist_but_missing"] + results["should_not_exist_but_found"]
                       + results["value_mismatch"] + results["row_or_col_not_in_slice"])

    if mismatch_details:
        print(f"\nFirst 20 mismatches:")
        for d in mismatch_details[:20]:
            print(d)
        pd.DataFrame(mismatch_details, columns=["row_acc", "col_acc", "issue"]).to_csv(args.out_mismatch, index=False)
        print(f"\nAll {len(mismatch_details)} mismatches saved to {args.out_mismatch}")
    else:
        print("\n[OK] Everything matches, no problems found")

    print(f"\nTotal problems: {total_problems} / {len(gt)}")


if __name__ == "__main__":
    main()