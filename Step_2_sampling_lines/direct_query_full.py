#!/usr/bin/env python3
"""
direct_query_full.py

read_pc_mat.py's CLI hardcodes top-10 display for --query_file mode
(process_query_file always does res['neighbor_ids'][:10]), even though the
underlying read_pc_mat_module.query() already returns the FULL neighbor
list in memory. This script imports the SAME module directly (no source
modification, just bypassing the print-only CLI wrapper) and dumps the full
neighbor list per accession to disk, in the same "ID,Jaccard" format that
query_pc_mat --show_all uses, so downstream comparison scripts can treat
both tools' outputs identically.

Usage:
    python3 direct_query_full.py \
        --matrix /scratch/mgs_project/matrix_unzipped \
        --db /scratch/mgs_project/db \
        --query_file sample2_query_accessions.txt \
        --module_dir /scratch/hvw5426/metagenome_vector_sketches/src \
        --out_dir decoded_py
"""

import argparse
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--module_dir", required=True,
                         help="directory containing read_pc_mat_module (same dir as read_pc_mat.py)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    sys.path.append(args.module_dir)
    import read_pc_mat_module as rpc  # same module read_pc_mat.py uses internally

    os.makedirs(args.out_dir, exist_ok=True)

    start = time.perf_counter()
    results = rpc.query(args.matrix, args.db, args.query_file)
    elapsed = time.perf_counter() - start
    print(f"Query completed in {elapsed:.6f} seconds.")

    for res in results:
        acc = res["id"]
        neighbor_ids = res["neighbor_ids"]
        jac = res["jaccard_similarities"]
        out_path = os.path.join(args.out_dir, f"{acc}.csv")
        with open(out_path, "w") as f:
            f.write("ID,Jaccard\n")
            for nid, j in zip(neighbor_ids, jac):
                f.write(f"{nid},{j}\n")

    print(f"Wrote {len(results)} full neighbor-list files to {args.out_dir}")


if __name__ == "__main__":
    main()