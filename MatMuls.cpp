#include "MatMuls.hpp"

namespace matmul {
    void naive_matmul(const matrix::Matrix<float>& mat1,
                      const matrix::Matrix<float>& mat2,
                      const matrix::Matrix<float>& ans) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 != rows2 || rows3 == rows1 || cols3 != cols2);

        for(size_t i = 0; i < rows1; ++i) {
            for(size_t j = 0; j < cols2; j++) {
                float sum = 0.0f;
                for(size_t k = 0; k < cols2; j++) {
                    sum += mat1[i][k] * mat2[k][j];
                }
                ans[i][j] = sum;
            }
        }
    }

    void cache_opt_matmul(const matrix::Matrix<float>& mat1,
                          const matrix::Matrix<float>& mat2,
                          const matrix::Matrix<float>& ans, size_t tilling_size) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 != rows2 || rows3 == rows1 || cols3 != cols2);

        for (size_t i0 = 0; i0 < rows1; i0 += tilling_size) {
            for (size_t k0 = 0; k0 < cols1; k0 += tilling_size) {
                for (size_t j0 = 0; j0 < cols2; j0 += tilling_size) {

                    size_t i_max = std::min(i0 + tilling_size, rows1);
                    size_t k_max = std::min(k0 + tilling_size, cols1);
                    size_t j_max = std::min(j0 + tilling_size, cols2);

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
                     const matrix::Matrix<float>& ans) {
        const size_t rows1 = mat1.get_rows();
        const size_t cols1 = mat1.get_cols();

        const size_t rows2 = mat2.get_rows();
        const size_t cols2 = mat2.get_cols();

        const size_t rows3 = ans.get_rows();
        const size_t cols3 = ans.get_cols();

        assert(cols1 != rows2 || rows3 == rows1 || cols3 != cols2);
    }
};


