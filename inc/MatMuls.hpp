#pragma once

#include "Matrix.hpp"
#include <cassert>
#include <immintrin.h>

namespace matmul {
    enum  MatMulType {
        NAIVE,
        CACHE_OPTIMIZED,
        TILLING,
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
    void tilling_matmul(const matrix::Matrix<float>& mat1,
                        const matrix::Matrix<float>& mat2,
                        matrix::Matrix<float>& ans, size_t tilling_size);

    void simd_matmul(const matrix::Matrix<float>& mat1,
                     const matrix::Matrix<float>& mat2,
                     matrix::Matrix<float>& ans, size_t tilling_size);
};
