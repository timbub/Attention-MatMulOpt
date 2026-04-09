#pragma once

#include "Matrix.hpp"
#include <cassert>

namespace matmul {
    enum  MatMulType {
        NAIVE,
        CACHE_OPTIMIZED,
        SIMD
    };

    void transpose(const matrix::Matrix<float>& src, matrix::Matrix<float>& dst);

    void scaling(matrix::Matrix<float>& mat, float scale);

    void naive_matmul(const matrix::Matrix<float>& mat1,
                      const matrix::Matrix<float>& mat2,
                      matrix::Matrix<float>& ans, size_t tilling_size);

    void cache_opt_matmul(const matrix::Matrix<float>& mat1,
                          const matrix::Matrix<float>& mat2,
                          matrix::Matrix<float>& ans, size_t tilling_size);

    void simd_matmul(const matrix::Matrix<float>& mat1,
                     const matrix::Matrix<float>& mat2,
                     matrix::Matrix<float>& ans, size_t tilling_size);
};
