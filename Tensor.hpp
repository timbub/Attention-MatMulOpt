#pragma once
#include "BufMatrix.hpp"
#include "Matrix.hpp"

namespace tensor {
    template <typename ElemT> class Tensor {
    private:
        size_t batch_size_;
        size_t seq_len_;
        size_t dim_;
        matrix::BufMatrix<ElemT> data_;
    public:
        Tensor(size_t b, size_t s, size_t d) : batch_size_(b), seq_len_(s), dim_(d) {}
        Tensor(const Tensor& other)            = delete;
        Tensor& operator=(const Tensor& other) = delete;

        Tensor(Tensor&& other) noexcept            = default;
        Tensor& operator=(Tensor&& other) noexcept = default;

        matrix::Matrix<ElemT> get_batch(size_t idx) const {
            return matrix::Matrix<ElemT>(seq_len_, dim_, const_cast<ElemT*>(data_.get_row_ptr(idx * seq_len_, dim_)));
        }

        size_t  get_batch_size() const {return batch_size_;};
        size_t  get_seq_len()    const {return seq_len_;}
        size_t  get_dim()        const {return dim_;}
    };
};

