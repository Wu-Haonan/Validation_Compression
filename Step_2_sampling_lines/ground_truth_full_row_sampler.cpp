// ground_truth_full_row_sampler.cpp
//
// Selects N (or 2N) accessions, then computes their FULL (unquantized)
// neighbor lists against ALL vectors in vectors.bin, replicating the exact
// math from pairwise_comp_optimized.cpp -- INCLUDING using Eigen for the
// dot product computation, matching production's
// `block_i.transpose() * block_j` approach. A naive triple-nested scalar
// loop is far slower than Eigen's internally vectorized/blocked matrix
// multiply, especially as N grows -- this version scales much better.
//
// Unlike Step_1 (which only needs specific row x col pairs via random
// access), this needs the FULL neighbor list per selected row, so we
// stream through vectors.bin ONCE sequentially in chunks, and for each
// chunk compute a (N x chunk_size) dot-product matrix against all N query
// vectors at once via Eigen GEMM.
//
// SELECTION MODES:
//   random  : N accessions, independently sampled (uniform random).
//   extreme : the N accessions with the SMALLEST vector norm and the N
//             accessions with the LARGEST vector norm (2N total query
//             rows). Each of these 2N rows is queried against the ENTIRE
//             database (not against each other only) -- this targets
//             whether vectors at the extremes of length/complexity
//             (very short/simple vs very long/complex sequences) behave
//             correctly as QUERY rows in realistic usage, which random
//             sampling is very unlikely to specifically hit.
//
// QUANTIZATION: quantization (round(raw*255)) is done HERE in C++ using
// std::round, the exact same function production code uses (see
// write_sparse_results_jaccard_wo_sort's `round(jaccard * MULT_CONST)`).
// Downstream Python scripts should READ the jaccard_quantized column, not
// recompute it with Python's round() (banker's rounding, can disagree
// with C++ at exact .5 boundaries).
//
// Usage:
//   g++ -std=c++17 -O3 -march=native -ffast-math -I<eigen_include> -o ground_truth_full_row_sampler ground_truth_full_row_sampler.cpp
//   ./ground_truth_full_row_sampler random  <db_folder> <N> <output_prefix> [seed]
//   ./ground_truth_full_row_sampler extreme <db_folder> <N> <output_prefix>
//
// Outputs (both modes):
//   <output_prefix>_query_accessions.txt   the selected accessions (one per line)
//   <output_prefix>_ground_truth/{accession}.csv   per-accession full neighbor
//       list, columns: col_acc,jaccard_raw,jaccard_quantized, SORTED
//       DESCENDING by jaccard_quantized (ties broken by ascending col_idx,
//       OUR OWN convention for reproducibility -- production's actual
//       tie-break rule inside pc_mat::query() uses std::sort with no
//       secondary key, so ties are not guaranteed any particular order;
//       this is why Top-K comparisons downstream use SET comparisons, not
//       strict order).

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>

namespace fs = std::filesystem;
using namespace std;
using namespace Eigen;

using MatrixXll = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

struct VectorDB {
    int dimension = 0;
    int total_vectors = 0;
    vector<string> accessions;
    vector<double> norms_sq;
    vector<double> norms;
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

// Computes full-row ground truth for the given query_indices against the
// ENTIRE database, via a single streaming pass + Eigen GEMM, and writes
// output files. Shared by both 'random' and 'extreme' modes.
void compute_and_write_full_rows(const string& db_folder_in, const VectorDB& db,
                                  vector<int> query_indices, const string& output_prefix) {
    string db_folder = db_folder_in;
    if (!db_folder.empty() && db_folder.back() != '/') db_folder += '/';
    string matrix_file = db_folder + "vectors.bin";
    uint64_t vector_bytes = static_cast<uint64_t>(db.dimension) * sizeof(int32_t);

    sort(query_indices.begin(), query_indices.end());
    int N = static_cast<int>(query_indices.size());

    // ---- Load the N query vectors into an Eigen matrix: dimension x N ----
    MatrixXll query_matrix(db.dimension, N);
    {
        ifstream vec_file(matrix_file, ios::binary);
        if (!vec_file) { cerr << "Cannot open vectors.bin\n"; exit(1); }
        vector<int32_t> buf(db.dimension);
        for (int qi = 0; qi < N; ++qi) {
            int idx = query_indices[qi];
            vec_file.seekg(static_cast<uint64_t>(idx) * vector_bytes);
            vec_file.read(reinterpret_cast<char*>(buf.data()), vector_bytes);
            if (!vec_file) { cerr << "Read failed for query idx " << idx << "\n"; exit(1); }
            for (int k = 0; k < db.dimension; ++k) query_matrix(k, qi) = buf[k];
        }
    }

    vector<vector<pair<int,double>>> neighbors(N);

    // ---- Single streaming sequential pass over vectors.bin, chunked, using Eigen GEMM ----
    {
        ifstream vec_file(matrix_file, ios::binary);
        if (!vec_file) { cerr << "Cannot open vectors.bin for streaming\n"; exit(1); }

        const int CHUNK = 4096;
        vector<int32_t> raw_buf(static_cast<size_t>(CHUNK) * db.dimension);

        int processed = 0;
        while (processed < db.total_vectors) {
            int this_chunk = min(CHUNK, db.total_vectors - processed);
            vec_file.read(reinterpret_cast<char*>(raw_buf.data()),
                          static_cast<uint64_t>(this_chunk) * vector_bytes);
            if (!vec_file && !vec_file.eof()) { cerr << "Streaming read failed\n"; exit(1); }

            MatrixXll cand_matrix(db.dimension, this_chunk);
            for (int c = 0; c < this_chunk; ++c) {
                for (int k = 0; k < db.dimension; ++k) {
                    cand_matrix(k, c) = raw_buf[static_cast<size_t>(c) * db.dimension + k];
                }
            }

            MatrixXll dot_products = query_matrix.transpose() * cand_matrix;

            for (int qi = 0; qi < N; ++qi) {
                int q_global_idx = query_indices[qi];
                double norm_q = db.norms_sq[q_global_idx];

                for (int c = 0; c < this_chunk; ++c) {
                    int cand_idx = processed + c;
                    if (cand_idx == q_global_idx) continue;  // self-pair, handled separately

                    double norm_c = db.norms_sq[cand_idx];
                    double threshold = 0.05 * (norm_q + norm_c);

                    int64_t dot = dot_products(qi, c);

                    // Existence check uses truncated integer division,
                    // exactly matching main()/compute_sparse_dot_products_optimized
                    // (see Step_1 history for the full rationale).
                    int64_t inter_truncated = dot / db.dimension;
                    bool should_exist = static_cast<double>(inter_truncated) > threshold;
                    if (!should_exist) continue;

                    double inter_full = static_cast<double>(dot) / db.dimension;
                    double jaccard_raw = inter_full / (norm_q + norm_c - inter_full);
                    if (jaccard_raw > 1) jaccard_raw = 1;

                    neighbors[qi].emplace_back(cand_idx, jaccard_raw);
                }
            }

            processed += this_chunk;
            if (processed % (CHUNK * 50) == 0 || processed == db.total_vectors) {
                cerr << "  streamed " << processed << " / " << db.total_vectors << " vectors\r";
            }
        }
        cerr << endl;
    }

    // ---- Write outputs ----
    ofstream qa_out(output_prefix + "_query_accessions.txt");
    for (int idx : query_indices) qa_out << db.accessions[idx] << "\n";
    qa_out.close();

    string gt_dir = output_prefix + "_ground_truth/";
    fs::create_directories(gt_dir);

    for (int qi = 0; qi < N; ++qi) {
        sort(neighbors[qi].begin(), neighbors[qi].end(),
             [](const pair<int,double>& a, const pair<int,double>& b) {
                 if (a.second != b.second) return a.second > b.second;
                 return a.first < b.first;
             });

        string acc = db.accessions[query_indices[qi]];
        ofstream out(gt_dir + acc + ".csv");
        out << "col_acc,jaccard_raw,jaccard_quantized\n";
        for (auto& [cand_idx, jac] : neighbors[qi]) {
            int quantized = static_cast<int>(round(jac * 255.0));
            out << db.accessions[cand_idx] << "," << jac << "," << quantized << "\n";
        }
    }

    cout << "Wrote ground truth full neighbor lists for " << N
         << " accessions to " << gt_dir << "\n";
    cout << "Query accession list: " << output_prefix << "_query_accessions.txt\n";
}

int run_random_mode(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " random <db_folder> <N> <output_prefix> [seed]\n";
        return 1;
    }
    string db_folder = argv[2];
    int N = stoi(argv[3]);
    string output_prefix = argv[4];
    unsigned int seed = (argc > 5) ? stoul(argv[5]) : 42;

    VectorDB db;
    if (!load_db(db_folder, db)) return 1;
    cout << "Total vectors: " << db.total_vectors << ", dimension: " << db.dimension << endl;
    if (N > db.total_vectors) { cerr << "Error: N exceeds total_vectors\n"; return 1; }

    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, db.total_vectors - 1);
    unordered_set<int> chosen_set;
    vector<int> query_indices;
    while ((int)query_indices.size() < N) {
        int idx = dist(rng);
        if (chosen_set.insert(idx).second) query_indices.push_back(idx);
    }

    cout << "Selected " << N << " random accessions.\n";
    compute_and_write_full_rows(db_folder, db, query_indices, output_prefix);
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
    if (2 * N > db.total_vectors) { cerr << "Error: 2*N exceeds total_vectors\n"; return 1; }

    // Sort indices by norm ascending
    vector<int> order(db.total_vectors);
    for (int i = 0; i < db.total_vectors; ++i) order[i] = i;
    sort(order.begin(), order.end(), [&](int a, int b) { return db.norms[a] < db.norms[b]; });

    vector<int> query_indices;
    query_indices.insert(query_indices.end(), order.begin(), order.begin() + N);          // N smallest
    query_indices.insert(query_indices.end(), order.end() - N, order.end());              // N largest
    // NOTE: each of these 2N accessions is queried as a row against the
    // ENTIRE database (compute_and_write_full_rows streams all
    // total_vectors candidates), NOT just against each other -- this is
    // different from Step_1's "extreme" mode, which builds a 2N x 2N
    // submatrix of JUST the extreme-norm vectors against each other.
    // Here we want to know how an extreme-norm vector behaves as a query
    // row in realistic usage (compared against the whole database), not
    // how extreme-norm vectors compare only amongst themselves.

    cout << "Selected " << N << " smallest-norm + " << N << " largest-norm accessions "
         << "(extreme mode), " << query_indices.size() << " total query rows.\n";
    cout << "Smallest norm in this set: " << db.norms[order[0]]
         << ", largest norm in this set: " << db.norms[order[db.total_vectors - 1]] << endl;

    compute_and_write_full_rows(db_folder, db, query_indices, output_prefix);
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage:\n"
             << "  " << argv[0] << " random  <db_folder> <N> <output_prefix> [seed]\n"
             << "  " << argv[0] << " extreme <db_folder> <N> <output_prefix>\n";
        return 1;
    }
    string mode = argv[1];
    if (mode == "random") return run_random_mode(argc, argv);
    if (mode == "extreme") return run_extreme_mode(argc, argv);
    cerr << "Unknown mode: " << mode << " (expected 'random' or 'extreme')\n";
    return 1;
}