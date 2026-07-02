#!/usr/bin/env python3
"""
topk_compare.py

Check #3/#4: for each sampled accession and each K in --ks, compare the
Top-K neighbor SET from ground truth vs decoded.

QUANTIZATION: this script does NOT quantize anything itself. It reads the
jaccard_quantized column directly from the ground truth CSV, which is
computed in C++ (ground_truth_full_row_sampler.cpp) using std::round --
the exact same rounding function production code uses. This avoids any
Python-vs-C++ rounding discrepancy at exact .5 boundaries (Python's
round() uses banker's rounding; C++'s std::round() rounds half away from
zero).

TIE DEFINITION: a "tie" is NOT "two raw jaccard values that happen to be
numerically close". A tie is "two neighbors whose raw jaccard values
quantize to the EXACT SAME 8-bit integer" -- because that is the only
situation where the compressed matrix genuinely cannot distinguish between
them (their decoded values would be byte-for-byte identical, not merely
close). If two raw values quantize to DIFFERENT integers, their decoded
values are necessarily different, the ranking between them is
well-defined, and getting it wrong IS a real bug -- no tolerance applies
there.

So for each row, given K:
  - rank ground-truth neighbors by their (C++-computed) quantized value
    descending; let q_boundary = the quantized value at rank K
  - "must_include" = every neighbor with q > q_boundary (unambiguous,
    strictly higher-ranked than the cutoff -- MUST appear in decoded Top-K)
  - "tie_pool"     = every neighbor with q == q_boundary (genuinely tied at
    the cutoff -- any subset of these filling the remaining slots is valid)
  - "ambiguous"    = neighbors whose RAW value is within 1/255 of the
    boundary's raw value, even if their quantized integer differs --
    absorbs upstream floating-point noise (production compiled with
    -ffast-math vs our own strict computation) that can shift a value's
    quantized bucket by +/-1 right at a rounding edge
  - anything outside must_include/tie_pool/ambiguous must NOT appear in
    decoded Top-K

We treat the production sort/tie-break logic as a black box and only check
the SET-level correctness implied by this quantization structure, not any
particular order within a tie group.

Usage:
    python3 topk_compare.py \
        --query_accessions sample2_query_accessions.txt \
        --gt_dir sample2_ground_truth \
        --decoded_dir decoded_cpp \
        --decoded_suffix _sample2.csv \
        --ks 10 20 50 100 1000 10000 \
        --out_mismatch topk_mismatches_cpp.csv
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
                          f"with the updated ground_truth_full_row_sampler.cpp (quantization "
                          f"now happens in C++, not Python).")
    # Sort by the C++-computed quantized value descending (NOT raw jaccard --
    # this is the same ranking key production code actually sorts by).
    df = df.sort_values("jaccard_quantized", ascending=False)
    return list(zip(df["col_acc"], df["jaccard_raw"], df["jaccard_quantized"]))


def load_decoded_sorted(decoded_dir, suffix, acc):
    filename = f"{acc}.csv" if suffix == "" else f"{acc}{suffix}"
    path = os.path.join(decoded_dir, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df[df["ID"] != acc]  # drop self-pair
    df = df.sort_values("Jaccard", ascending=False)
    return list(zip(df["ID"], df["Jaccard"]))


def topk_check(gt_list, dec_list, k):
    """
    gt_list:  [(acc, jaccard_raw, jaccard_quantized), ...] from load_gt_sorted,
              already sorted descending by jaccard_quantized.
    dec_list: [(acc, jaccard_decoded), ...] -- full decoded list, sorted
              descending by the caller (or already truncated to K by the
              tool itself, in which case len(dec_list) <= k).
    """
    gt_raw_dict = {acc: raw for acc, raw, q in gt_list}
    gt_q_dict = {acc: q for acc, raw, q in gt_list}

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

        # AMBIGUOUS ZONE: any accession whose raw ground-truth value is
        # within QUANT_TOLERANCE of the boundary's raw value -- absorbs
        # upstream floating-point noise affecting boundary determination
        # itself, not just value comparisons.
        ambiguous = set(acc for acc, raw, q in gt_list
                         if abs(raw - boundary_raw) <= QUANT_TOLERANCE)

        must_include = set(acc for acc, raw, q in gt_list if q > q_boundary) - ambiguous
        tie_pool = (set(acc for acc, raw, q in gt_list if q == q_boundary) | ambiguous) - must_include

    dec_topk = dec_list[:k]
    dec_set = set(x[0] for x in dec_topk)

    # (1) every unambiguous, strictly-higher-ranked neighbor MUST be present
    missing_must = must_include - dec_set
    if missing_must:
        return "fail", f"missing required higher-ranked neighbors (not a tie issue): {missing_must}"

    # (2) decoded shouldn't have MORE entries than expected slots allow when
    #     gt has fewer than k real neighbors
    if n_gt <= k and len(dec_set) > n_gt:
        extra = dec_set - must_include
        return "fail", f"decoded has extra neighbors beyond ground truth's full list: {extra}"

    # (3) nothing outside must_include/tie_pool/ambiguous should appear
    allowed = must_include | tie_pool
    leaked = dec_set - allowed
    if leaked:
        return "fail", f"decoded Top-{k} contains neighbors ranked below the boundary (q < {q_boundary}): {leaked}"

    # (4) value check for must_include entries: compare against the RAW
    #     (unquantized) ground truth, with 1/255 tolerance -- NOT
    #     zero-tolerance against our own quantized value. See module
    #     docstring for why (upstream -ffast-math floating-point noise).
    dec_vals = dict(dec_topk)
    for acc in must_include:
        expected_raw = gt_raw_dict[acc]
        actual = dec_vals.get(acc)
        if actual is None or abs(expected_raw - actual) > QUANT_TOLERANCE:
            return "fail", f"value mismatch for unambiguous neighbor {acc}: expected_raw={expected_raw}, actual={actual}"

    # If decoded's exact slot-filling from the tie pool matches gt's own
    # quantized-order Top-K choice -> exact; otherwise it's a legitimate tie
    gt_topk_set = set(acc for acc, raw, q in gt_list[:k])
    if dec_set == gt_topk_set:
        return "exact", "exact set match"
    return "tie", (f"set differs only within the genuine tie pool at the boundary "
                    f"(quantized value {q_boundary}/255) -- order among tied neighbors is ambiguous")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_accessions", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--decoded_dir", required=True)
    parser.add_argument("--decoded_suffix", default="",
                         help="suffix INCLUDING .csv, e.g. '_sample2.csv'. Empty means {acc}.csv.")
    parser.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS)
    parser.add_argument("--out_mismatch", default="topk_mismatches.csv")
    args = parser.parse_args()

    with open(args.query_accessions) as f:
        accs = [line.strip() for line in f if line.strip()]

    results = {k: {"exact": 0, "tie": 0, "fail": 0} for k in args.ks}
    mismatches = []

    for acc in accs:
        gt_list = load_gt_sorted(args.gt_dir, acc)
        dec_list = load_decoded_sorted(args.decoded_dir, args.decoded_suffix, acc)

        if dec_list is None:
            if len(gt_list) == 0:
                # EXPECTED: see full_row_compare.py's note -- pc_mat::query()
                # writes no file for genuinely zero-neighbor rows.
                for k in args.ks:
                    results[k]["exact"] += 1
                continue
            for k in args.ks:
                results[k]["fail"] += 1
                mismatches.append((acc, k, "decoded file missing (but ground truth has neighbors -- real miss)"))
            continue

        for k in args.ks:
            status, msg = topk_check(gt_list, dec_list, k)
            results[k][status] += 1
            if status == "fail":
                mismatches.append((acc, k, msg))

    print("===== Top-K validation results =====")
    for k in args.ks:
        r = results[k]
        total_pass = r["exact"] + r["tie"]
        print(f"Top-{k}: pass={total_pass} (exact={r['exact']}, tie-tolerance={r['tie']}), fail={r['fail']}")

    if mismatches:
        pd.DataFrame(mismatches, columns=["accession", "k", "issue"]).to_csv(args.out_mismatch, index=False)
        print(f"\n{len(mismatches)} mismatches saved to {args.out_mismatch}")
        print("\nFirst 20 mismatches:")
        for m in mismatches[:20]:
            print(m)
    else:
        print("\n[OK] all Top-K checks passed")


if __name__ == "__main__":
    main()