#include <iostream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include "Attention.hpp"


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
        std::cerr << "Usage: " << argv[0] << " <naive|cache|simd> [tilling_size]\n";
        return 1;
    }

    std::string mode = argv[1];
    matmul::MatMulType matmul_type;

    size_t tilling_size = 32;

    if (mode == "naive") {
        matmul_type = matmul::MatMulType::NAIVE;
    } else if (mode == "cache") {
        matmul_type = matmul::MatMulType::CACHE_OPTIMIZED;
    } else if (mode == "tilling") {
        matmul_type = matmul::MatMulType::TILLING;

        if (argc >= 3) {
            tilling_size = std::stoul(argv[2]);
        }
    } else if (mode == "simd") {
        matmul_type = matmul::MatMulType::SIMD;
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'\n";
        return 1;
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

        auto start_time = std::chrono::high_resolution_clock::now();
        tensor::Tensor<float> result = attention::attention_with_matmul(Q, K, V, matmul_type, tilling_size);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> time = end_time - start_time;

        std::cout << "[MODE: " << mode << "] | Attention time: " << time.count() << " ms\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
