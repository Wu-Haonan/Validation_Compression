#!/usr/bin/env python3
"""
topk_function_check.py

Validates the --top K feature ITSELF (the production sorting/truncation
logic inside pc_mat::query() / read_pc_mat_module), as opposed to
topk_compare.py (which validates Top-K via --show_all + our own re-sort).

This script does NOT re-sort anything itself. It takes the raw output of
running the tools with --top K directly, and checks it against the same
quantization-based tie definition as topk_compare.py: a "tie" means two
neighbors quantize to the EXACT SAME 8-bit integer (their decoded values
would be byte-for-byte identical), not merely "close" raw values. See
topk_compare.py's module docstring for the full rationale.

QUANTIZATION: this script does NOT quantize anything itself either. It
reads the jaccard_quantized column directly from the ground truth CSV
(computed in C++ via std::round, matching production exactly) -- see
topk_compare.py for why this matters (Python's round() uses banker's
rounding and can disagree with C++'s std::round() at exact .5 boundaries).

If this disagrees with topk_compare.py's results (which validates the same
K values via --show_all + manual re-sort), that specifically isolates a bug
in the --top K parameter's own truncation logic, separate from any
correctness issue in the underlying stored values themselves.

NOTE: self-pairs are stored in the matrix and always rank #1 with
jaccard=1 (see Step_1 findings). When invoking query_pc_mat for this
check, you must request --top (K+1) so that after self is stripped here,
exactly K real neighbors remain to compare fairly against ground truth's K
real neighbors -- see run_step2.sh for how this is handled.

Usage:
    python3 topk_function_check.py \
        --query_accessions sample2_query_accessions.txt \
        --gt_dir sample2_ground_truth \
        --decoded_base_dir decoded_top_cpp \
        --decoded_suffix _sample2.csv \
        --ks 10 20 50 100 1000 10000 \
        --out_mismatch topk_function_mismatches_cpp.csv
"""

import argparse
import os
import pandas as pd

QUANT_LEVELS = 255
QUANT_TOLERANCE = 1.0 / 255.0  # tolerance against RAW ground truth (not quantized)
DEFAULT_KS = [10, 20, 50, 100, 1000, 10000]


def load_gt_sorted(gt_dir, acc):
    path = os.path.join(gt_dir, f"{acc}.csv")
    df = pd.read_csv(path)
    if "jaccard_quantized" not in df.columns:
        raise ValueError(f"{path} is missing 'jaccard_quantized' -- regenerate ground truth "
                          f"with the updated ground_truth_full_row_sampler.cpp.")
    df = df.sort_values("jaccard_quantized", ascending=False)
    return list(zip(df["col_acc"], df["jaccard_raw"], df["jaccard_quantized"]))


def load_decoded_topk(decoded_dir_for_k, suffix, acc):
    """Load the --top (K+1) output AS-IS (no re-sorting), then drop self --
    this is exactly what the tool produced for this K, modulo the self-pair
    slot we requested extra to account for."""
    filename = f"{acc}.csv" if suffix == "" else f"{acc}{suffix}"
    path = os.path.join(decoded_dir_for_k, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["ID"] != acc]  # drop self-pair
    return list(zip(df["ID"], df["Jaccard"]))


def topk_check(gt_list, dec_topk, k):
    """Same logic as topk_compare.py's topk_check. gt_list here is
    [(acc, jaccard_raw, jaccard_quantized), ...] already sorted descending
    by jaccard_quantized. dec_topk is taken as-is from the tool's own
    --top K output (already truncated by the tool itself, modulo self-pair
    handling), not re-derived by us from a full list."""
    gt_raw_dict = {acc: raw for acc, raw, q in gt_list}

    n_gt = len(gt_list)
    if n_gt == 0:
        must_include, tie_pool, q_boundary, ambiguous = set(), set(), None, set()
    elif n_gt <= k:
        must_include = set(acc for acc, raw, q in gt_list)
        tie_pool = set()
        q_boundary = gt_list[-1][2]
        ambiguous = set()
    else:
        boundary_acc, boundary_raw, q_boundary = gt_list[k - 1]
        ambiguous = set(acc for acc, raw, q in gt_list
                         if abs(raw - boundary_raw) <= QUANT_TOLERANCE)
        must_include = set(acc for acc, raw, q in gt_list if q > q_boundary) - ambiguous
        tie_pool = (set(acc for acc, raw, q in gt_list if q == q_boundary) | ambiguous) - must_include

    dec_set = set(x[0] for x in dec_topk)

    missing_must = must_include - dec_set
    if missing_must:
        return "fail", f"--top {k} is missing required higher-ranked neighbors (not a tie issue): {missing_must}"

    if n_gt <= k and len(dec_set) > n_gt:
        extra = dec_set - must_include
        return "fail", f"--top {k} returned extra neighbors beyond ground truth's full list: {extra}"

    allowed = must_include | tie_pool
    leaked = dec_set - allowed
    if leaked:
        return "fail", f"--top {k} contains neighbors ranked below the boundary (q < {q_boundary}): {leaked}"

    # Value check: compare against RAW ground truth with 1/255 tolerance
    # (see topk_compare.py for the full rationale).
    dec_vals = dict(dec_topk)
    for acc in must_include:
        expected_raw = gt_raw_dict[acc]
        actual = dec_vals.get(acc)
        if actual is None or abs(expected_raw - actual) > QUANT_TOLERANCE:
            return "fail", f"value mismatch for unambiguous neighbor {acc}: expected_raw={expected_raw}, actual={actual}"

    gt_topk_set = set(acc for acc, raw, q in gt_list[:k])
    if dec_set == gt_topk_set:
        return "exact", "exact set match"
    return "tie", (f"set differs only within the genuine tie pool at the boundary "
                    f"(quantized value {q_boundary}/255) -- order among tied neighbors is ambiguous")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_accessions", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--decoded_base_dir", required=True,
                         help="base dir containing one subfolder per K, named top_{k}/ "
                              "(e.g. decoded_top_cpp/top_10/, decoded_top_cpp/top_1000/)")
    parser.add_argument("--decoded_suffix", default="",
                         help="suffix INCLUDING .csv, e.g. '_top10.csv'. Empty means {acc}.csv.")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--out_mismatch", default="topk_function_mismatches.csv")
    args = parser.parse_args()

    with open(args.query_accessions) as f:
        accs = [line.strip() for line in f if line.strip()]

    results = {k: {"exact": 0, "tie": 0, "fail": 0} for k in args.ks}
    mismatches = []

    for k in args.ks:
        decoded_dir_for_k = os.path.join(args.decoded_base_dir, f"top_{k}")
        for acc in accs:
            gt_list = load_gt_sorted(args.gt_dir, acc)
            dec_topk = load_decoded_topk(decoded_dir_for_k, args.decoded_suffix, acc)

            if dec_topk is None:
                if len(gt_list) == 0:
                    results[k]["exact"] += 1
                    continue
                results[k]["fail"] += 1
                mismatches.append((acc, k, f"decoded file missing in {decoded_dir_for_k} (but ground truth has neighbors -- real miss)"))
                continue

            status, msg = topk_check(gt_list, dec_topk, k)
            results[k][status] += 1
            if status == "fail":
                mismatches.append((acc, k, msg))

    print("===== --top K function validation results =====")
    for k in args.ks:
        r = results[k]
        total_pass = r["exact"] + r["tie"]
        print(f"--top {k}: pass={total_pass} (exact={r['exact']}, tie-tolerance={r['tie']}), fail={r['fail']}")

    if mismatches:
        pd.DataFrame(mismatches, columns=["accession", "k", "issue"]).to_csv(args.out_mismatch, index=False)
        print(f"\n{len(mismatches)} mismatches saved to {args.out_mismatch}")
        print("\nFirst 20 mismatches:")
        for m in mismatches[:20]:
            print(m)
    else:
        print("\n[OK] all --top K function checks passed")


if __name__ == "__main__":
    main()