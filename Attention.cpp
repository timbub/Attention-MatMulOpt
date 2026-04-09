#include "Attention.hpp"

namespace attention {

    void softmax(matrix::Matrix<float>& m) {
        const size_t rows = m.get_rows();
        const size_t cols = m.get_cols();

        for (size_t i = 0; i < rows; ++i) {

            float max_val = m[i][0];
            for (size_t j = 1; j < cols; ++j) {
                if (m[i][j] > max_val) {
                    max_val = m[i][j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                m[i][j] = std::exp(m[i][j] - max_val);
                sum_exp += m[i][j];
            }

            float inv_sum = 1.0f / sum_exp;
            for (size_t j = 0; j < cols; ++j) {
                m[i][j] *= inv_sum;
            }
        }
    }

    tensor::Tensor<float> attention_with_matmul(const tensor::Tensor<float>& Q,
                                                const tensor::Tensor<float>& K,
                                                const tensor::Tensor<float>& V,
                                                matmul::MatMulType matmul_type,
                                                size_t tilling_size) {
        const size_t batch_size = Q.get_batch_size();
        const size_t seq_q = Q.get_seq_len();
        const size_t seq_k = K.get_seq_len();
        const size_t dk = Q.get_dim();
        const size_t dv = V.get_dim();

        tensor::Tensor<float> result(batch_size, seq_q, dv);
        matrix::Matrix<float> scores(seq_q, seq_k);
        matrix::Matrix<float> k_T(dk, seq_k);

        float scale = 1.0f / std::sqrt(static_cast<float>(dk));

        using MatMulFuncPtr = void (*)(const matrix::Matrix<float>&, const matrix::Matrix<float>&, matrix::Matrix<float>&, size_t);
        MatMulFuncPtr run_matmul = nullptr;

        switch (matmul_type) {
            case matmul::MatMulType::NAIVE:           run_matmul = matmul::naive_matmul; break;
            case matmul::MatMulType::CACHE_OPTIMIZED: run_matmul = matmul::cache_opt_matmul; break;
            case matmul::MatMulType::SIMD:            run_matmul = matmul::simd_matmul;  break;
        }

        for (size_t i = 0; i < batch_size; ++i) {
            auto q_mat = Q.get_batch(i);
            auto k_mat = K.get_batch(i);
            auto v_mat = V.get_batch(i);
            auto res_mat = result.get_batch(i);

            matmul::transpose(k_mat, k_T);
            run_matmul(q_mat, k_T, scores, matmul_type);
            matmul::scaling(scores, scale);
            softmax(scores);

            run_matmul(scores, v_mat, res_mat, matmul_type);
        }
        return result;
    }
};
