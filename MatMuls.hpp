#pragma once

#include "Matrix.hpp"

namespace matmul {
    enum  MatMulType {
        NAIVE,
        CACHE_OPTIMIZED,
        SIMD
    };

    void naive_matmul(const matrix::Matrix<float>& mat1,
                      const matrix::Matrix<float>& mat2,
                      const matrix::Matrix<float>& ans);

    void cache_opt_matmul(const matrix::Matrix<float>& mat1,
                          const matrix::Matrix<float>& mat2,
                          const matrix::Matrix<float>& ans);

    void simd_matmul(const matrix::Matrix<float>& mat1,
                     const matrix::Matrix<float>& mat2,
                     const matrix::Matrix<float>& ans);
};
