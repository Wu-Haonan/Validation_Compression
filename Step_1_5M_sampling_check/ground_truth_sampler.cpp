// ground_truth_sampler.cpp
//
// Independently re-implements the math logic from pairwise_comp_optimized.cpp
// (dot product -> threshold -> raw jaccard -> quantized jaccard), used to
// generate ground truth for validating the compressed matrix.
//
// QUANTIZATION NOTE: quantization (round(raw*255)) is done HERE in C++,
// using the exact same std::round() that production code uses (see
// write_sparse_results_jaccard_wo_sort's `round(jaccard * MULT_CONST)`),
// rather than being re-derived in Python downstream. This avoids any
// language-level rounding discrepancy (e.g. Python's banker's-rounding
// round() vs C++'s round-half-away-from-zero std::round()) at exact .5
// boundaries -- the quantized_jaccard column in the output CSV should be
// treated as authoritative; Python scripts should READ it, not recompute it.
//
// SAMPLING MODES:
//   random  : N rows x N cols, independently sampled (uniform random),
//             disjoint sets -- as before.
//   extreme : the N accessions with the SMALLEST vector norm and the N
//             accessions with the LARGEST vector norm (2N total), used as
//             BOTH rows and cols, producing a full (2N x 2N) submatrix.
//             This targets vectors at the extremes of length/complexity
//             (e.g. very short/simple vs very long/complex sequences),
//             whose dot-product/threshold behavior may differ from
//             "typical" randomly-sampled vectors and is unlikely to be
//             hit by chance under random sampling.
//
// Usage:
//   g++ -std=c++17 -O3 -march=native -ffast-math -o ground_truth_sampler ground_truth_sampler.cpp
//   ./ground_truth_sampler random  <db_folder> <num_rows> <num_cols> <output_prefix> [seed]
//   ./ground_truth_sampler extreme <db_folder> <N>                  <output_prefix>
//
// Outputs (both modes):
//   <output_prefix>_ground_truth.csv   row_idx,col_idx,row_acc,col_acc,should_exist,jaccard_raw,jaccard_quantized
//   <output_prefix>_row_file.txt       accessions for --row_file
//   <output_prefix>_col_file.txt       accessions for --col_file

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <unordered_set>
#include <algorithm>
#include <cmath>

using namespace std;

struct VectorDB {
    int dimension = 0;
    int total_vectors = 0;
    vector<string> accessions;
    vector<double> norms_sq;   // squared norm, matches main()'s norm*norm convention
    vector<double> norms;      // raw norm, needed for extreme-sampling sort
};

bool load_db(const string& db_folder_in, VectorDB& db) {
    string db_folder = db_folder_in;
    if (!db_folder.empty() && db_folder.back() != '/') db_folder += '/';

    {
        ifstream dim_in(db_folder + "dimension.txt");
        if (!dim_in) { cerr << "Cannot read dimension.txt\n"; return false; }
        dim_in >> db.dimension;
    }
    {
        ifstream norms_in(db_folder + "vector_norms.txt");
        if (!norms_in) { cerr << "Cannot read vector_norms.txt\n"; return false; }
        string acc; double norm;
        while (norms_in >> acc >> norm) {
            db.accessions.push_back(acc);
            db.norms.push_back(norm);
            db.norms_sq.push_back(norm * norm);
        }
    }
    db.total_vectors = static_cast<int>(db.accessions.size());
    return db.total_vectors > 0 && db.dimension > 0;
}

bool read_vector(ifstream& vec_file, int idx, int dimension, vector<int32_t>& buf) {
    uint64_t vector_bytes = static_cast<uint64_t>(dimension) * sizeof(int32_t);
    buf.resize(dimension);
    vec_file.seekg(static_cast<uint64_t>(idx) * vector_bytes);
    vec_file.read(reinterpret_cast<char*>(buf.data()), vector_bytes);
    return static_cast<bool>(vec_file);
}

const double MULT_CONST = static_cast<double>((1ULL << 8) - 1);  // 255, matches production

void compute_and_write_pair(ofstream& gt_out, ifstream& vec_file, const VectorDB& db,
                             int i, int j, vector<int32_t>& vi, vector<int32_t>& vj) {
    if (i == j) return;  // self-pairs excluded, matching ground_truth_full_row_sampler.cpp's convention

    if (!read_vector(vec_file, i, db.dimension, vi)) { cerr << "Read failed at idx " << i << "\n"; exit(1); }
    if (!read_vector(vec_file, j, db.dimension, vj)) { cerr << "Read failed at idx " << j << "\n"; exit(1); }

    int64_t dot = 0;
    for (int k = 0; k < db.dimension; ++k) {
        dot += static_cast<int64_t>(vi[k]) * static_cast<int64_t>(vj[k]);
    }

    double norm_i = db.norms_sq[i];
    double norm_j = db.norms_sq[j];
    double threshold = 0.05 * (norm_i + norm_j);

    // Existence check: matches main()'s integer-truncated division exactly
    // (see Step_1/Step_2 history for the detailed rationale on this).
    int64_t inter_truncated = dot / db.dimension;
    bool should_exist = static_cast<double>(inter_truncated) > threshold;

    double jaccard_raw = -1.0;
    int quantized = -1;
    if (should_exist) {
        double inter_full = static_cast<double>(dot) / db.dimension;
        jaccard_raw = inter_full / (norm_i + norm_j - inter_full);
        if (jaccard_raw > 1) jaccard_raw = 1;
        // Quantize HERE, using std::round (round-half-away-from-zero),
        // exactly matching write_sparse_results_jaccard_wo_sort's
        // `round(jaccard * MULT_CONST)`.
        quantized = static_cast<int>(round(jaccard_raw * MULT_CONST));
    }

    gt_out << i << "," << j << "," << db.accessions[i] << "," << db.accessions[j]
           << "," << (should_exist ? 1 : 0) << "," << jaccard_raw << "," << quantized << "\n";
}

void write_accession_list(const string& path, const VectorDB& db, const vector<int>& indices) {
    ofstream out(path);
    for (int idx : indices) out << db.accessions[idx] << "\n";
}

int run_random_mode(int argc, char* argv[]) {
    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " random <db_folder> <num_rows> <num_cols> <output_prefix> [seed]\n";
        return 1;
    }
    string db_folder = argv[2];
    int num_rows = stoi(argv[3]);
    int num_cols = stoi(argv[4]);
    string output_prefix = argv[5];
    unsigned int seed = (argc > 6) ? stoul(argv[6]) : 42;

    VectorDB db;
    if (!load_db(db_folder, db)) return 1;
    cout << "Total vectors: " << db.total_vectors << ", dimension: " << db.dimension << endl;

    if (num_rows + num_cols > db.total_vectors) {
        cerr << "Error: num_rows + num_cols exceeds total_vectors\n"; return 1;
    }

    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, db.total_vectors - 1);
    unordered_set<int> chosen;
    vector<int> row_indices, col_indices;
    while ((int)row_indices.size() < num_rows) {
        int idx = dist(rng);
        if (chosen.insert(idx).second) row_indices.push_back(idx);
    }
    while ((int)col_indices.size() < num_cols) {
        int idx = dist(rng);
        if (chosen.insert(idx).second) col_indices.push_back(idx);
    }

    string matrix_file = db_folder;
    if (!matrix_file.empty() && matrix_file.back() != '/') matrix_file += '/';
    matrix_file += "vectors.bin";
    ifstream vec_file(matrix_file, ios::binary);
    if (!vec_file) { cerr << "Cannot open vectors.bin\n"; return 1; }

    ofstream gt_out(output_prefix + "_ground_truth.csv");
    gt_out << "row_idx,col_idx,row_acc,col_acc,should_exist,jaccard_raw,jaccard_quantized\n";

    vector<int32_t> vi, vj;
    for (int i : row_indices) {
        for (int j : col_indices) {
            compute_and_write_pair(gt_out, vec_file, db, i, j, vi, vj);
        }
    }

    write_accession_list(output_prefix + "_row_file.txt", db, row_indices);
    write_accession_list(output_prefix + "_col_file.txt", db, col_indices);

    cout << "Sampled " << num_rows << " rows x " << num_cols << " cols (random mode).\n";
    return 0;
}

int run_extreme_mode(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " extreme <db_folder> <N> <output_prefix>\n";
        return 1;
    }
    string db_folder = argv[2];
    int N = stoi(argv[3]);
    string output_prefix = argv[4];

    VectorDB db;
    if (!load_db(db_folder, db)) return 1;
    cout << "Total vectors: " << db.total_vectors << ", dimension: " << db.dimension << endl;

    if (2 * N > db.total_vectors) {
        cerr << "Error: 2*N exceeds total_vectors\n"; return 1;
    }

    // Sort indices by norm ascending
    vector<int> order(db.total_vectors);
    for (int i = 0; i < db.total_vectors; ++i) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) { return db.norms[a] < db.norms[b]; });

    vector<int> smallest(order.begin(), order.begin() + N);
    vector<int> largest(order.end() - N, order.end());

    vector<int> combined;
    combined.insert(combined.end(), smallest.begin(), smallest.end());
    combined.insert(combined.end(), largest.begin(), largest.end());
    // combined now has 2N indices: N smallest-norm + N largest-norm vectors,
    // used as BOTH rows and cols -> full (2N x 2N) submatrix.

    string matrix_file = db_folder;
    if (!matrix_file.empty() && matrix_file.back() != '/') matrix_file += '/';
    matrix_file += "vectors.bin";
    ifstream vec_file(matrix_file, ios::binary);
    if (!vec_file) { cerr << "Cannot open vectors.bin\n"; return 1; }

    ofstream gt_out(output_prefix + "_ground_truth.csv");
    gt_out << "row_idx,col_idx,row_acc,col_acc,should_exist,jaccard_raw,jaccard_quantized\n";

    vector<int32_t> vi, vj;
    for (int i : combined) {
        for (int j : combined) {
            compute_and_write_pair(gt_out, vec_file, db, i, j, vi, vj);
        }
    }

    write_accession_list(output_prefix + "_row_file.txt", db, combined);
    write_accession_list(output_prefix + "_col_file.txt", db, combined);

    cout << "Sampled " << N << " smallest-norm + " << N << " largest-norm accessions "
         << "(extreme mode), full " << combined.size() << "x" << combined.size() << " submatrix.\n";
    cout << "Smallest norm in this set: " << db.norms[smallest.front()]
         << ", largest norm in this set: " << db.norms[largest.back()] << endl;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " random  <db_folder> <num_rows> <num_cols> <output_prefix> [seed]\n"
             << "  " << argv[0] << " extreme <db_folder> <N> <output_prefix>\n";
        return 1;
    }
    string mode = argv[1];
    if (mode == "random") return run_random_mode(argc, argv);
    if (mode == "extreme") return run_extreme_mode(argc, argv);
    cerr << "Unknown mode: " << mode << " (expected 'random' or 'extreme')\n";
    return 1;
}