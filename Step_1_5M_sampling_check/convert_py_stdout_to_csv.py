#!/usr/bin/env python3
"""
convert_py_stdout_to_csv.py

read_pc_mat.py has no --write_to_file option -- it only prints to stdout.
Looking at its actual source (process_row_col() in read_pc_mat.py):

    print(f"Processing row_file: {row_file}, col_file: {col_file} in {matrix_folder}")
    ...
    print(f"Query completed in {elapsed:.6f} seconds.\n")
    ...
    print(df.to_string())

So the captured stdout contains TWO extra lines before the actual table
(the "Processing row_file..." line and the "Query completed..." line, plus a
blank line), followed by the pivoted table itself:

             col_acc_1   col_acc_2   ...
row_acc_1     0.505882    0.00000
row_acc_2     0.000000    0.74902
...

This script strips out those leading non-table lines (and any blank lines),
then parses the remaining space-aligned table into a real CSV.

Usage:
    python3 convert_py_stdout_to_csv.py --input sample1_slice_py_raw.txt --output sample1_slice_py.csv
"""

import argparse
import io
import pandas as pd

# Prefixes of the known non-table lines printed by read_pc_mat.py before the table
SKIP_PREFIXES = ("Processing row_file:", "Query completed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="raw captured stdout from read_pc_mat.py")
    parser.add_argument("--output", required=True, help="output csv path")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    table_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        if any(stripped.startswith(p) for p in SKIP_PREFIXES):
            continue
        table_lines.append(line)

    if not table_lines:
        raise ValueError(f"No table content found in {args.input} after filtering -- "
                          f"check whether read_pc_mat.py's output format has changed.")

    table_text = "".join(table_lines)

    # sep=r'\s+' handles the variable-width space alignment in the printed table.
    # index_col=0 treats the first column (row accessions) as the row index.
    df = pd.read_csv(io.StringIO(table_text), sep=r"\s+", index_col=0)
    df.to_csv(args.output)
    print(f"Converted {args.input} -> {args.output}, shape={df.shape}")


if __name__ == "__main__":
    main()