// ground_truth_for_specific_rows.cpp
//
// Like Step_2's ground_truth_full_row_sampler.cpp, but instead of randomly
// SAMPLING N accessions, takes an EXPLICIT list of target accessions (read
// from a file) and computes their full, unquantized neighbor lists against
// ALL vectors in this db_folder. Reuses the same Eigen-accelerated single
// streaming pass over vectors.bin, and the same exact-truncation math as
// pairwise_comp_optimized.cpp (see Step_1/Step_2 for the detailed
// rationale on the integer-truncation behavior in the threshold check).
//
// This is sized for the real 35k-accession subset -- O(N_targets x 35000 x
// dimension), NOT O(35000 x 35000), which would be far too large.
//
// Usage:
//   g++ -std=c++17 -O2 -I<eigen_include> -o ground_truth_for_specific_rows ground_truth_for_specific_rows.cpp
//   ./ground_truth_for_specific_rows <db_folder> <target_accessions_file> <output_prefix>
//
// Outputs:
//   <output_prefix>_ground_truth/{accession}.csv   per-target full neighbor
//       list, columns: col_acc,jaccard_raw, SORTED DESCENDING by jaccard_raw
//       (ties broken by ascending col_idx -- see Step_2's note on why this
//       is OUR convention, not necessarily the production tie-break rule)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <Eigen/Dense>

namespace fs = std::filesystem;
using namespace std;
using namespace Eigen;
using MatrixXll = Eigen::Matrix<int64_t, Eigen::Dynamic, Eigen::Dynamic>;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Usage: " << argv[0] << " <db_folder> <target_accessions_file> <output_prefix>\n";
        return 1;
    }
    string db_folder = argv[1];
    string target_file = argv[2];
    string output_prefix = argv[3];

    if (!db_folder.empty() && db_folder.back() != '/') db_folder += '/';

    int dimension = 0;
    {
        ifstream dim_in(db_folder + "dimension.txt");
        if (!dim_in) { cerr << "Cannot read dimension.txt\n"; return 1; }
        dim_in >> dimension;
    }

    vector<string> accessions;
    vector<double> all_norms_sq;
    unordered_map<string,int> acc_to_idx;
    {
        ifstream norms_in(db_folder + "vector_norms.txt");
        if (!norms_in) { cerr << "Cannot read vector_norms.txt\n"; return 1; }
        string acc; double norm;
        int idx = 0;
        while (norms_in >> acc >> norm) {
            accessions.push_back(acc);
            all_norms_sq.push_back(norm * norm);
            acc_to_idx[acc] = idx++;
        }
    }
    int total_vectors = static_cast<int>(accessions.size());
    cout << "Total vectors: " << total_vectors << ", dimension: " << dimension << endl;

    // ---- Read target accessions, resolve to indices ----
    vector<int> query_indices;
    {
        ifstream tf(target_file);
        if (!tf) { cerr << "Cannot read target_accessions_file\n"; return 1; }
        string line;
        while (tf >> line) {
            auto it = acc_to_idx.find(line);
            if (it == acc_to_idx.end()) {
                cerr << "Warning: target accession '" << line << "' not found in this db_folder, skipping\n";
                continue;
            }
            query_indices.push_back(it->second);
        }
    }
    int N = static_cast<int>(query_indices.size());
    cout << "Resolved " << N << " target accessions\n";
    if (N == 0) { cerr << "No valid targets, aborting.\n"; return 1; }

    string matrix_file = db_folder + "vectors.bin";
    uint64_t vector_bytes = static_cast<uint64_t>(dimension) * sizeof(int32_t);

    // ---- Load the N query vectors into an Eigen matrix: dimension x N ----
    MatrixXll query_matrix(dimension, N);
    {
        ifstream vec_file(matrix_file, ios::binary);
        if (!vec_file) { cerr << "Cannot open vectors.bin\n"; return 1; }
        vector<int32_t> buf(dimension);
        for (int qi = 0; qi < N; ++qi) {
            int idx = query_indices[qi];
            vec_file.seekg(static_cast<uint64_t>(idx) * vector_bytes);
            vec_file.read(reinterpret_cast<char*>(buf.data()), vector_bytes);
            if (!vec_file) { cerr << "Read failed for query idx " << idx << "\n"; return 1; }
            for (int k = 0; k < dimension; ++k) query_matrix(k, qi) = buf[k];
        }
    }

    vector<vector<pair<int,double>>> neighbors(N);

    // ---- Single streaming sequential pass over vectors.bin, chunked, using Eigen GEMM ----
    {
        ifstream vec_file(matrix_file, ios::binary);
        if (!vec_file) { cerr << "Cannot open vectors.bin for streaming\n"; return 1; }

        const int CHUNK = 4096;
        vector<int32_t> raw_buf(static_cast<size_t>(CHUNK) * dimension);

        int processed = 0;
        while (processed < total_vectors) {
            int this_chunk = min(CHUNK, total_vectors - processed);
            vec_file.read(reinterpret_cast<char*>(raw_buf.data()),
                          static_cast<uint64_t>(this_chunk) * vector_bytes);
            if (!vec_file && !vec_file.eof()) { cerr << "Streaming read failed\n"; return 1; }

            MatrixXll cand_matrix(dimension, this_chunk);
            for (int c = 0; c < this_chunk; ++c) {
                for (int k = 0; k < dimension; ++k) {
                    cand_matrix(k, c) = raw_buf[static_cast<size_t>(c) * dimension + k];
                }
            }

            MatrixXll dot_products = query_matrix.transpose() * cand_matrix;

            for (int qi = 0; qi < N; ++qi) {
                int q_global_idx = query_indices[qi];
                double norm_q = all_norms_sq[q_global_idx];

                for (int c = 0; c < this_chunk; ++c) {
                    int cand_idx = processed + c;
                    if (cand_idx == q_global_idx) continue;  // self-pair excluded

                    double norm_c = all_norms_sq[cand_idx];
                    double threshold = 0.05 * (norm_q + norm_c);

                    int64_t dot = dot_products(qi, c);
                    int64_t inter_truncated = dot / dimension;
                    bool should_exist = static_cast<double>(inter_truncated) > threshold;
                    if (!should_exist) continue;

                    double inter_full = static_cast<double>(dot) / dimension;
                    double jaccard_raw = inter_full / (norm_q + norm_c - inter_full);
                    if (jaccard_raw > 1) jaccard_raw = 1;

                    neighbors[qi].emplace_back(cand_idx, jaccard_raw);
                }
            }

            processed += this_chunk;
            if (processed % (CHUNK * 50) == 0 || processed == total_vectors) {
                cerr << "  streamed " << processed << " / " << total_vectors << " vectors\r";
            }
        }
        cerr << endl;
    }

    string gt_dir = output_prefix + "_ground_truth/";
    fs::create_directories(gt_dir);

    for (int qi = 0; qi < N; ++qi) {
        sort(neighbors[qi].begin(), neighbors[qi].end(),
             [](const pair<int,double>& a, const pair<int,double>& b) {
                 if (a.second != b.second) return a.second > b.second;
                 return a.first < b.first;
             });

        string acc = accessions[query_indices[qi]];
        ofstream out(gt_dir + acc + ".csv");
        out << "col_acc,jaccard_raw\n";
        for (auto& [cand_idx, jac] : neighbors[qi]) {
            out << accessions[cand_idx] << "," << jac << "\n";
        }
    }

    cout << "Wrote ground truth full neighbor lists for " << N
         << " target accessions to " << gt_dir << "\n";

    return 0;
}