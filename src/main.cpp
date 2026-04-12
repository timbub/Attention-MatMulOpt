#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <sys/resource.h>
#include "Attention.hpp"

enum AttType {
    BASIC,
    FLASH
};

long long get_peak_rss_kib() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

tensor::Tensor<float> read_tensor_from_cin() {
    size_t batch, seq, dim;
    if (!(std::cin >> batch >> seq >> dim)) {
        throw std::runtime_error("Failed to read dimensions for tensor");
    }

    tensor::Tensor<float> tensor(batch, seq, dim);

    for (size_t b = 0; b < batch; ++b) {
        matrix::Matrix<float> mat = tensor.get_batch(b);
        for (size_t i = 0; i < seq; ++i) {
            for (size_t j = 0; j < dim; ++j) {
                std::cin >> mat[i][j];
            }
        }
    }
    return tensor;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <b|f> <naive|cache|simd|tiling> [tiling_size]\n";
        return 1;
    }

    std::string type = argv[1];
    AttType at_type;
    if (type == "b") {
        at_type = AttType::BASIC;
    } else if (type == "f") {
        at_type = AttType::FLASH;
    } else {
        std::cerr << "Error: Unknown type '" << type << "'\n";
        return 1;
    }

    matmul::MatMulType matmul_type = matmul::MatMulType::CACHE_OPTIMIZED; // дефолт
    std::string mode;

    if (at_type == AttType::BASIC) {
        if (argc < 3) {
            std::cerr << "Error: Basic attention requires a matmul mode (naive|cache|simd|tiling)\n";
            return 1;
        }
        mode = argv[2];
        if (mode == "naive") matmul_type = matmul::MatMulType::NAIVE;
        else if (mode == "cache") matmul_type = matmul::MatMulType::CACHE_OPTIMIZED;
        else if (mode == "tiling") matmul_type = matmul::MatMulType::TILLING;
        else if (mode == "simd") matmul_type = matmul::MatMulType::SIMD;
        else {
            std::cerr << "Error: Unknown mode '" << mode << "'\n";
            return 1;
        }
    }

    size_t tiling_size = 32;
    if (argc >= 4) {
        tiling_size = std::stoul(argv[3]);
    }

    try {
        tensor::Tensor<float> Q = read_tensor_from_cin();
        tensor::Tensor<float> K = read_tensor_from_cin();
        tensor::Tensor<float> V = read_tensor_from_cin();

        if (Q.get_batch_size() != K.get_batch_size() || K.get_batch_size() != V.get_batch_size()) {
            throw std::invalid_argument("Batch sizes must be equal");
        }
        if (Q.get_dim() != K.get_dim()) {
            throw std::invalid_argument("Dim of Q and K must be equal (dk)");
        }
        if (K.get_seq_len() != V.get_seq_len()) {
            throw std::invalid_argument("Sequence length of K and V must be equal");
        }

        tensor::Tensor<float> result(Q.get_batch_size(), Q.get_seq_len(), Q.get_dim());

        auto start_time = std::chrono::high_resolution_clock::now();
        if (at_type == AttType::BASIC) {
            result = attention::attention_with_matmul(Q, K, V, matmul_type, tiling_size);
        } else {
            result = attention::flash_attention(Q, K, V, tiling_size);
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time = end_time - start_time;

        if (at_type == AttType::BASIC) {
            std::cout << "[BASIC | MODE: " << mode << "] Time: " << time.count() << " ms\n";
        } else {
            std::cout << "[FLASH]" << "Time: " << time.count() << " ms\n";
        }

        std::cout << "Peak Memory: " << get_peak_rss_kib() / 1024 << " MB\n";

        std::cerr << "Done. (Check: " << result.get_batch(0)[0][0] << ")\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
