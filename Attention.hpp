#pragma once
#include "Tensor.hpp"
#include "MatMuls.hpp"

namespace attention {

    tensor::Tensor<float> attention_with_matmul(const tensor::Tensor<float>& Q,
                                                const tensor::Tensor<float>& K,
                                                const tensor::Tensor<float>& V) {
        
    }
};
