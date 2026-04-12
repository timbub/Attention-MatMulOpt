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
                                                size_t tiling_size) {
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
            case matmul::MatMulType::TILLING:         run_matmul = matmul::tiling_matmul;  break;
            case matmul::MatMulType::SIMD:            run_matmul = matmul::simd_matmul;  break;
        }

        for (size_t i = 0; i < batch_size; ++i) {
            auto q_mat = Q.get_batch(i);
            auto k_mat = K.get_batch(i);
            auto v_mat = V.get_batch(i);
            auto res_mat = result.get_batch(i);

            std::fill(&scores[0][0], &scores[0][0] + (seq_q * seq_k), 0.0f);
            matmul::transpose(k_mat, k_T);
            run_matmul(q_mat, k_T, scores, tiling_size);
            matmul::scaling(scores, scale);
            softmax(scores);

            run_matmul(scores, v_mat, res_mat, tiling_size);
        }
        return result;
    }

    tensor::Tensor<float> flash_attention(const tensor::Tensor<float>& Q,
                                          const tensor::Tensor<float>& K,
                                          const tensor::Tensor<float>& V,
                                          size_t tiling_size) {
        const size_t batch_size = Q.get_batch_size();
        const size_t seq_q = Q.get_seq_len();
        const size_t seq_k = K.get_seq_len();
        const size_t dk = Q.get_dim();
        const size_t dv = V.get_dim();

        tensor::Tensor<float> result(batch_size, seq_q, dv);
        matrix::Matrix<float> k_T(dk, seq_k);

        const float scale = 1.0f / std::sqrt(static_cast<float>(dk));

        std::vector<float> S_block(tiling_size * tiling_size, 0.0f); //for Q*K_t
        std::vector<float> O_block(tiling_size * dv, 0.0f); //for S_block*V
        std::vector<float> m_val(tiling_size, -INFINITY);
        std::vector<float> d_val(tiling_size, 0.0f);

        std::vector<float> Q_scaled(tiling_size * dk, 0.0f);

        for (size_t b = 0; b < batch_size; ++b) {
            auto q_mat = Q.get_batch(b);
            auto k_mat = K.get_batch(b);
            auto v_mat = V.get_batch(b);
            auto out_mat = result.get_batch(b);

            matmul::transpose(k_mat, k_T);

            for (size_t i0 = 0; i0 < seq_q; i0 += tiling_size) {
                size_t i_max = std::min(i0 + tiling_size, seq_q);
                size_t block_q_len = i_max - i0;

                std::fill(m_val.begin(), m_val.begin() + block_q_len, -INFINITY);
                std::fill(d_val.begin(), d_val.begin() + block_q_len, 0.0f);
                std::fill(O_block.begin(), O_block.begin() + (block_q_len * dv), 0.0f);

                //Q*scale
                for (size_t i = 0; i < block_q_len; ++i) {
                    const float* q_row = &q_mat[i0 + i][0];
                    float* q_s_row = &Q_scaled[i * dk];
                    for (size_t k = 0; k < dk; ++k) {
                        q_s_row[k] = q_row[k] * scale;
                    }
                }

                //create S_block
                for (size_t j0 = 0; j0 < seq_k; j0 += tiling_size) {
                    size_t j_max = std::min(j0 + tiling_size, seq_k);
                    size_t block_k_len = j_max - j0;

                    std::fill(S_block.begin(), S_block.begin() + (block_q_len * block_k_len), 0.0f);

                    for (size_t i = 0; i < block_q_len; ++i) {
                        const float* q_row = &Q_scaled[i * dk];
                        float*  s_row = &S_block[i * block_k_len];

                        for (size_t k = 0; k < dk; ++k) {
                            float r = q_row[k];
                            const float* kt_row = &k_T[k][j0];
                            for (size_t j = 0; j < block_k_len; ++j) {
                                s_row[j] += r * kt_row[j];
                            }
                        }
                    }

                    //softmax
                    for (size_t i = 0; i < block_q_len; ++i) {
                        float*  s_row = &S_block[i * block_k_len];
                        float*  o_row = &O_block[i * dv];

                        float m_block = -INFINITY;
                        for (size_t j = 0; j < block_k_len; ++j) {
                            if (s_row[j] > m_block) m_block = s_row[j];
                        }

                        float m_old = m_val[i];
                        float m_new = std::max(m_old, m_block);
                        float exp_old = (m_old == -INFINITY) ? 0.0f : std::exp(m_old - m_new);

                        float d_block = 0.0f;
                        for (size_t j = 0; j < block_k_len; ++j) {
                            s_row[j] = std::exp(s_row[j] - m_new);
                            d_block += s_row[j];
                        }

                        m_val[i] = m_new;
                        d_val[i] = d_val[i] * exp_old + d_block;

                        for (size_t v = 0; v < dv; ++v) {
                            o_row[v] *= exp_old;
                        }

                        //create  O_block
                        for (size_t j = 0; j < block_k_len; ++j) {
                            float p = s_row[j];
                            const float* v_row = &v_mat[j0 + j][0];
                            for (size_t v = 0; v < dv; ++v) {
                                o_row[v] += p * v_row[v];
                            }
                        }
                    }
                }
                //answer = answer/d
                for (size_t i = 0; i < block_q_len; ++i) {
                    float inv_d = 1.0f / d_val[i];
                    const float* o_row = &O_block[i * dv];
                    float* out_row = &out_mat[i0 + i][0];

                    for (size_t v = 0; v < dv; ++v) {
                        out_row[v] = o_row[v] * inv_d;
                    }
                }
            }
        }
        return result;
    }
};
