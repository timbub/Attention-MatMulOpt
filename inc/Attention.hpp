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


};
