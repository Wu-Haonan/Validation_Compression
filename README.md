# Step 1– Pair-level spot check

Method: sample random(smallest/largest) N rows × N cols, query both the C++ and Python tools against the compressed matrix (/scratch/mgs_project/matrix_unzipped), then independently compute the true Jaccard from vectors.bin (/scratch/mgs_project/db). Accept if the difference is within 1/255 (the quantization step size).

## Modify `run_step1.sh`

Modify the configs at the top of the file

```bash
# ===== Config, edit as needed =====
DB_FOLDER="/scratch/mgs_project/db"
MATRIX_FOLDER="/scratch/mgs_project/matrix_unzipped"
QUERY_PC_MAT_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/query_pc_mat"
READ_PC_MAT_PY="/scratch/hvw5426/metagenome_vector_sketches/src/read_pc_mat.py"
EIGEN_INCLUDE="/scratch/hvw5426/metagenome_vector_sketches/include/Eigen"
NUM_ROWS=100
NUM_COLS=100
EXTREME_N=50          # extreme mode: N smallest-norm + N largest-norm accessions
SEED=42
OUTPUT_PREFIX="sample1"
EXTREME_PREFIX="sample1_extreme"
THREADS=8
# ===================================
```

## Run `run_step1.sh`

```
./run_step1.sh
```



# Step 2 – Full-row & Top-K check

Method: sample random(smallest/largest) N accessions, retrieve their full rows via both tools, and check Jaccard values plus Top-K correctness (K = 10, 20, 50, 100, 1000, 10000). Note: Top-K results can differ when several neighbors are tied at the cutoff value. To distinguish real errors from ties, the check verifies whether the different accessions‘ values are tied with the K-th boundary element in ground truth.

## Modify `run_step2.sh`

Modify the configs at the top of the file

```bash
# ===== Config, edit as needed =====
DB_FOLDER="/scratch/mgs_project/db"
MATRIX_FOLDER="/scratch/mgs_project/matrix_unzipped"
QUERY_PC_MAT_BIN="/scratch/hvw5426/metagenome_vector_sketches/build/query_pc_mat"
READ_PC_MAT_MODULE_DIR="/scratch/hvw5426/metagenome_vector_sketches/src"
EIGEN_INCLUDE="/scratch/hvw5426/metagenome_vector_sketches/include/Eigen"
N=50
EXTREME_N=25           # extreme mode: N smallest-norm + N largest-norm accessions
SEED=42
OUTPUT_PREFIX="sample2"
EXTREME_PREFIX="sample2_extreme"
THREADS=8
KS="10 20 50 100 1000 10000"
# ===================================
```

## Run `run_step2.sh`

```
./run_step2.sh
```

