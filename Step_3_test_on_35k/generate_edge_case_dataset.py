#!/usr/bin/env python3
"""
generate_edge_case_dataset.py

Generates a small, fully-controlled synthetic dataset of DNA sequences for
Step_3 edge-case validation, using an INDEL mutation model (insertions and
deletions, no substitutions).

MUTATION RATE -> JACCARD CALIBRATION
There is no simple closed-form formula relating indel rate to k-mer set
Jaccard (unlike the pure-substitution Mash-distance model), because
indels shift all downstream sequence content rather than affecting
k-mers independently. Instead of guessing a formula, we EMPIRICALLY
CALIBRATE: for a candidate indel rate, generate several (base, mutant)
pairs, compute their TRUE k-mer set Jaccard directly in Python (exact
k-mer extraction, not sketched/projected), and binary-search the rate
until the average Jaccard is close to the target.

KNOWN APPROXIMATION GAP: this calibration uses the EXACT k-mer set
Jaccard from the raw sequences. Your production pipeline instead
estimates Jaccard from a sourmash sketch (FracMinHash subsampling) that
is then random-projected into a fixed-dimension vector -- both steps
introduce their own estimation noise on top of the true k-mer Jaccard.
So calibrated targets are a best-effort starting point, not a guarantee
of the exact value the real pipeline will report -- especially for the
"boundary_threshold" group, which is deliberately spread across several
nearby targets so that, empirically, some real pipeline outputs should
land on each side of the actual 5% dot-product threshold.

Usage:
    python3 generate_edge_case_dataset.py \
        --output_fasta step3_dataset.fasta \
        --output_manifest step3_manifest.csv \
        --kmer_size 21 \
        --seq_length 5000 \
        --seed 42
"""

import argparse
import csv
import random

BASES = "ACGT"


def random_sequence(length, rng):
    return "".join(rng.choice(BASES) for _ in range(length))


def indel_mutate(seq, rate, rng, indel_size_max=3):
    """Pure indel mutation model: at each original-sequence position, with
    probability `rate`, perform either an insertion or a deletion (50/50)
    of a random length in [1, indel_size_max]. No substitutions."""
    out = []
    i = 0
    n = len(seq)
    while i < n:
        if rng.random() < rate:
            if rng.random() < 0.5:
                # deletion: skip a random-length chunk
                del_len = rng.randint(1, indel_size_max)
                i += del_len
                continue
            else:
                # insertion: splice in random bases before the current base
                ins_len = rng.randint(1, indel_size_max)
                out.append(random_sequence(ins_len, rng))
        out.append(seq[i])
        i += 1
    return "".join(out)


def kmer_set(seq, k):
    if len(seq) < k:
        return set()
    return set(seq[i:i + k] for i in range(len(seq) - k + 1))


def true_kmer_jaccard(seq1, seq2, k):
    s1, s2 = kmer_set(seq1, k), kmer_set(seq2, k)
    if not s1 and not s2:
        return 0.0
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union > 0 else 0.0


def calibrate_mutation_rate(target_jaccard, k, seq_length, rng,
                             trials=5, max_iter=20, lo=0.0, hi=0.5):
    """Binary search for an indel rate whose average TRUE k-mer Jaccard
    (over `trials` random pairs) is close to target_jaccard. Jaccard
    decreases monotonically (on average) as rate increases."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        avg_j = 0.0
        for _ in range(trials):
            base = random_sequence(seq_length, rng)
            mut = indel_mutate(base, mid, rng)
            avg_j += true_kmer_jaccard(base, mut, k)
        avg_j /= trials
        if avg_j > target_jaccard:
            lo = mid  # not mutated enough yet, still too similar -> need higher rate
        else:
            hi = mid
    return (lo + hi) / 2


def write_fasta_entry(f, seq_id, seq):
    f.write(f">{seq_id}\n")
    for i in range(0, len(seq), 70):
        f.write(seq[i:i + 70] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_fasta", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--kmer_size", type=int, default=21,
                         help="MUST match the k-mer size your sourmash sketches actually use.")
    parser.add_argument("--seq_length", type=int, default=5000)
    parser.add_argument("--n_background", type=int, default=30)
    parser.add_argument("--n_boundary", type=int, default=15)
    parser.add_argument("--background_jaccard", type=float, default=0.35)
    parser.add_argument("--single_neighbor_jaccard", type=float, default=0.4)
    parser.add_argument("--near_identical_jaccard", type=float, default=0.97)
    parser.add_argument("--boundary_target_jaccard", type=float, default=0.0526,
                         help="theoretical center: the ~5%% dot-product threshold's "
                              "implied jaccard floor (see Step_1/2 discussion)")
    parser.add_argument("--boundary_spread", type=float, default=0.02)
    parser.add_argument("--calibration_trials", type=int, default=5,
                         help="random pairs averaged per binary-search step "
                              "(more = slower but less noisy calibration)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    k = args.kmer_size
    L = args.seq_length

    manifest_rows = []
    fasta_entries = []

    def add_pair(base_id, target_jaccard, group):
        rate = calibrate_mutation_rate(target_jaccard, k, L, rng, trials=args.calibration_trials)
        base_seq = random_sequence(L, rng)
        mut_seq = indel_mutate(base_seq, rate, rng)
        achieved_j = true_kmer_jaccard(base_seq, mut_seq, k)  # actual realization, not just the calibration average
        fasta_entries.append((base_id, base_seq))
        fasta_entries.append((base_id + "_mut", mut_seq))
        manifest_rows.append([base_id, group, "base", target_jaccard, rate, achieved_j, ""])
        manifest_rows.append([base_id + "_mut", group, "mutant", target_jaccard, rate, achieved_j, base_id])
        return achieved_j

    # ---- row0 marker: first entry in the file ----
    row0_seq = random_sequence(L, rng)
    fasta_entries.append(("row0_marker", row0_seq))
    manifest_rows.append(["row0_marker", "row0_marker", "isolated_random", "", "", "", ""])

    # ---- single neighbor pair: isolated, exactly 1 neighbor each ----
    j1 = add_pair("single_neighbor_A", args.single_neighbor_jaccard, "single_neighbor")

    # ---- zero neighbor: isolated singleton, no mutant, no relation to anyone ----
    zero_seq = random_sequence(L, rng)
    fasta_entries.append(("zero_neighbor_X", zero_seq))
    manifest_rows.append(["zero_neighbor_X", "zero_neighbor", "isolated_random", "", "", "", ""])

    # ---- near-identical pair: target jaccard near quantized max ----
    j2 = add_pair("near_identical_A", args.near_identical_jaccard, "near_identical")

    # ---- boundary-threshold group: spread around the ~5.26% theoretical floor ----
    boundary_js = []
    for i in range(args.n_boundary):
        frac = i / max(1, args.n_boundary - 1)
        target_j = args.boundary_target_jaccard - args.boundary_spread + 2 * args.boundary_spread * frac
        target_j = max(0.001, target_j)
        achieved = add_pair(f"boundary_{i:02d}", target_j, "boundary_threshold")
        boundary_js.append(achieved)

    # ---- background filler pairs: typical/moderate similarity ----
    for j in range(args.n_background):
        add_pair(f"background_{j:02d}", args.background_jaccard, "background")

    # ---- last row marker: appended last ----
    last_seq = random_sequence(L, rng)
    fasta_entries.append(("last_row_marker", last_seq))
    manifest_rows.append(["last_row_marker", "last_row_marker", "isolated_random", "", "", "", ""])

    with open(args.output_fasta, "w") as f:
        for seq_id, seq in fasta_entries:
            write_fasta_entry(f, seq_id, seq)

    with open(args.output_manifest, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["accession", "group", "role", "target_jaccard",
                          "indel_rate", "achieved_true_kmer_jaccard", "paired_with"])
        writer.writerows(manifest_rows)

    print(f"Wrote {len(fasta_entries)} sequences to {args.output_fasta}")
    print(f"Wrote manifest to {args.output_manifest}")
    print(f"k-mer size used for calibration: {k} (VERIFY this matches your sourmash sketches)")
    print(f"single_neighbor achieved true k-mer jaccard: {j1:.4f} (target {args.single_neighbor_jaccard})")
    print(f"near_identical achieved true k-mer jaccard:  {j2:.4f} (target {args.near_identical_jaccard})")
    print(f"boundary group achieved true k-mer jaccards:  "
          f"{', '.join(f'{x:.4f}' for x in boundary_js)}")


if __name__ == "__main__":
    main()