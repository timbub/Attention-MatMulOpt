#include "MatMuls.hpp"

namespace matmul {
    void transpose(const matrix::Matrix<float>& src, matrix::Matrix<float>& dst) {
        const size_t rows = src.get_rows();
        const size_t cols = src.get_cols();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                dst[j][i] = src[i][j];
            }
        }
    }

    void scaling(matrix::Matrix<float>& mat, float scale) {
        const size_t rows = mat.get_rows();
        const size_t cols = mat.get_cols();

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                mat[i][j] *= scale;
            }
        }
    }

    void naive_matmul(const matrix::Matrix<float>& mat1,
                      const matrix::Matrix<float>& mat2,
                      matrix::Matrix<float>& ans, size_t tiling_size) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 == rows2 && rows3 == rows1 && cols3 == cols2);

        for(size_t i = 0; i < rows1; ++i) {
            for(size_t j = 0; j < cols2; j++) {
                float sum = 0.0f;
                for(size_t k = 0; k < cols1; k++) {
                    sum += mat1[i][k] * mat2[k][j];
                }
                ans[i][j] = sum;
            }
        }
    }

    void cache_opt_matmul(const matrix::Matrix<float>& mat1,
                          const matrix::Matrix<float>& mat2,
                          matrix::Matrix<float>& ans, size_t tiling_size) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 == rows2 && rows3 == rows1 && cols3 == cols2);

        for (size_t i = 0; i < rows1; ++i) {
            for (size_t k = 0; k < cols1; ++k) {
                float r = mat1[i][k];
                for (size_t j = 0; j < cols2; ++j) {
                    ans[i][j] += r * mat2[k][j];
                }
            }
        }
    }
    void tiling_matmul(const matrix::Matrix<float>& mat1,
                        const matrix::Matrix<float>& mat2,
                        matrix::Matrix<float>& ans, size_t tiling_size) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 == rows2 && rows3 == rows1 && cols3 == cols2);

        for (size_t i0 = 0; i0 < rows1; i0 += tiling_size) {
            for (size_t k0 = 0; k0 < cols1; k0 += tiling_size) {
                for (size_t j0 = 0; j0 < cols2; j0 += tiling_size) {

                    size_t i_max = std::min(i0 + tiling_size, rows1);
                    size_t k_max = std::min(k0 + tiling_size, cols1);
                    size_t j_max = std::min(j0 + tiling_size, cols2);

                    for (size_t i = i0; i < i_max; ++i) {
                        for (size_t k = k0; k < k_max; ++k) {
                            float r = mat1[i][k];
                            for (size_t j = j0; j < j_max; ++j) {
                                ans[i][j] += r * mat2[k][j];
                            }
                        }
                    }

                }
            }
        }
    }

    void simd_matmul(const matrix::Matrix<float>& mat1,
                     const matrix::Matrix<float>& mat2,
                     matrix::Matrix<float>& ans, size_t tiling_size) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();
        const size_t cols2 = mat2.get_cols();

        for (size_t i0 = 0; i0 < rows1; i0 += tiling_size) {
            for (size_t j0 = 0; j0 < cols2; j0 += tiling_size) {
                for (size_t k0 = 0; k0 < cols1; k0 += tiling_size) {

                    size_t i_max = std::min(i0 + tiling_size, rows1);
                    size_t j_max = std::min(j0 + tiling_size, cols2);
                    size_t k_max = std::min(k0 + tiling_size, cols1);

                    size_t i = i0;
                    for (; i + 1 < i_max; i += 2) {
                        float* c_ptr0 = &ans[i][0];
                        float* c_ptr1 = &ans[i + 1][0];

                        size_t j = j0;
                        for (; j + 31 < j_max; j += 32) {

                            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
                            __m256 c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();

                            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
                            __m256 c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();

                            for (size_t k = k0; k < k_max; ++k) {
                                __m256 a0 = _mm256_set1_ps(mat1[i][k]);
                                __m256 a1 = _mm256_set1_ps(mat1[i + 1][k]);

                                const float* b_ptr = &mat2[k][j];
                                __m256 b0 = _mm256_loadu_ps(b_ptr);
                                __m256 b1 = _mm256_loadu_ps(b_ptr + 8);
                                __m256 b2 = _mm256_loadu_ps(b_ptr + 16);
                                __m256 b3 = _mm256_loadu_ps(b_ptr + 24);

                                c00 = _mm256_fmadd_ps(a0, b0, c00);
                                c01 = _mm256_fmadd_ps(a0, b1, c01);
                                c02 = _mm256_fmadd_ps(a0, b2, c02);
                                c03 = _mm256_fmadd_ps(a0, b3, c03);

                                c10 = _mm256_fmadd_ps(a1, b0, c10);
                                c11 = _mm256_fmadd_ps(a1, b1, c11);
                                c12 = _mm256_fmadd_ps(a1, b2, c12);
                                c13 = _mm256_fmadd_ps(a1, b3, c13);
                            }

                            _mm256_storeu_ps(c_ptr0 + j,      _mm256_add_ps(_mm256_loadu_ps(c_ptr0 + j), c00));
                            _mm256_storeu_ps(c_ptr0 + j + 8,  _mm256_add_ps(_mm256_loadu_ps(c_ptr0 + j + 8), c01));
                            _mm256_storeu_ps(c_ptr0 + j + 16, _mm256_add_ps(_mm256_loadu_ps(c_ptr0 + j + 16), c02));
                            _mm256_storeu_ps(c_ptr0 + j + 24, _mm256_add_ps(_mm256_loadu_ps(c_ptr0 + j + 24), c03));

                            _mm256_storeu_ps(c_ptr1 + j,      _mm256_add_ps(_mm256_loadu_ps(c_ptr1 + j), c10));
                            _mm256_storeu_ps(c_ptr1 + j + 8,  _mm256_add_ps(_mm256_loadu_ps(c_ptr1 + j + 8), c11));
                            _mm256_storeu_ps(c_ptr1 + j + 16, _mm256_add_ps(_mm256_loadu_ps(c_ptr1 + j + 16), c12));
                            _mm256_storeu_ps(c_ptr1 + j + 24, _mm256_add_ps(_mm256_loadu_ps(c_ptr1 + j + 24), c13));
                        }

                        for (; j < j_max; ++j) {
                            float sum0 = 0, sum1 = 0;
                            for (size_t k = k0; k < k_max; ++k) {
                                sum0 += mat1[i][k] * mat2[k][j];
                                sum1 += mat1[i + 1][k] * mat2[k][j];
                            }
                            c_ptr0[j] += sum0;
                            c_ptr1[j] += sum1;
                        }
                    }

                    for (; i < i_max; ++i) {
                        float* c_ptr = &ans[i][0];
                        for (size_t j = j0; j < j_max; ++j) {
                            float sum = 0;
                            for (size_t k = k0; k < k_max; ++k) {
                                sum += mat1[i][k] * mat2[k][j];
                            }
                            c_ptr[j] += sum;
                        }
                    }

                }
            }
        }
    }
};


