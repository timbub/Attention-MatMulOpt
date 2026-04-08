#pragma once
#include "Tensor.hpp"
#include "MatMuls.hpp"
#include "Matrix.hpp"

namespace attention {

    void softmax(matrix::Matrix<float>& m) {
        const size_t rows = m.get_rows();
        const size_t cols = m.get_cols();

        for (size_t i = 0; i < rows; ++i) {

            float max_val = scores[i][0];
            for (size_t j = 1; j < cols; ++j) {
                if (scores[i][j] > max_val) {
                    max_val = scores[i][j];
                }
            }

            float sum_exp = 0.0f;
            for (size_t j = 0; j < cols; ++j) {
                scores[i][j] = std::exp(scores[i][j] - max_val);
                sum_exp += scores[i][j];
            }

            float inv_sum = 1.0f / sum_exp;
            for (size_t j = 0; j < cols; ++j) {
                scores[i][j] *= inv_sum;
            }
        }
    }

    tensor::Tensor<float> attention_with_matmul(const tensor::Tensor<float>& Q,
                                                const tensor::Tensor<float>& K,
                                                const tensor::Tensor<float>& V, MatMulType matmul_type) {
        const size_t batch_size = Q.batch_size();
        const size_t seq_q = Q.seq_len();
        const size_t seq_k = K.seq_len();
        const size_t dk = Q.dim();
        const size_t dv = V.dim();

        Tensor result(batch_size, seq_q, dv);

        matrix::Matrix<float> scores(seq_q, seq_k);

        float scale = 1.0f / std::sqrt(static_cast<float>(dk));

        for (size_t i = 0; i < batch_size; ++i) {
            auto q_mat = Q.get_batch(b);
            auto k_mat = K.get_batch(b);
            auto v_mat = V.get_batch(b);
            auto res_mat = result.get_batch(b);

            //TODO add tranpose, scaling
            matmul(q_mat, transpose(k_mat), scores, matmul_type);
            scaling(scores, scale);
            softmax(scores);

            matmul(scores, v_mat, res_mat, matmul_type);
        }
        return result;
    }
};
