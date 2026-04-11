#pragma once
#include "Tensor.hpp"
#include "MatMuls.hpp"
#include "Matrix.hpp"

namespace attention {

    void softmax(matrix::Matrix<float>& m);

    tensor::Tensor<float> attention_with_matmul(const tensor::Tensor<float>& Q,
                                                const tensor::Tensor<float>& K,
                                                const tensor::Tensor<float>& V,
                                                matmul::MatMulType matmul_type,
                                                 size_t tilling_size);
    void process_flash_block(const matrix::Matrix<float>& Q,
                         const matrix::Matrix<float>& K,
                         const matrix::Matrix<float>& V,
                         matrix::Matrix<float>& out,
                         std::vector<float>& m,
                         std::vector<float>& d,
                         size_t j_start, size_t j_end);

    tensor::Tensor<float> flash_attention(const tensor::Tensor<float>& Q,
                                          const tensor::Tensor<float>& K,
                                          const tensor::Tensor<float>& V,
                                          size_t tilling_size);

};
