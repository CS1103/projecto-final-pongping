#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#pragma once

#include <vector>
#include <array>
#include <initializer_list>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <type_traits>
#include <tuple>
#include <utility>
#include <functional>

namespace utec {
namespace algebra {

template<typename T, size_t N>
class Tensor {
private:
    std::vector<T> data_;
    std::array<size_t, N> shape_{};
    std::array<size_t, N> strides_{};

    void compute_strides() {
        if (N > 0) {
            strides_[N-1] = 1;
            for (int i = int(N)-2; i >= 0; --i)
                strides_[i] = strides_[i+1] * shape_[i+1];
        }
    }

public:
    Tensor() {
        shape_.fill(0);
        strides_.fill(0);
    }

    explicit Tensor(const std::array<size_t, N>& shape)
      : shape_(shape) {
        size_t total = 1;
        for (auto d : shape_) total *= d;
        data_.resize(total);
        compute_strides();
    }

    // 2D convenience constructor
    template<typename U = T>
    Tensor(size_t r, size_t c, typename std::enable_if<N==2, U>::type* = nullptr)
      : Tensor(std::array<size_t,2>{r,c}) {}

    // copy and assign
    Tensor(const Tensor& o) = default;
    Tensor& operator=(const Tensor& o) = default;

    // initializer-list assign
    Tensor& operator=(std::initializer_list<T> vals) {
        assert(vals.size() == data_.size());
        std::copy(vals.begin(), vals.end(), data_.begin());
        return *this;
    }

    // element access for 2D
    template<size_t M = N>
    typename std::enable_if<M==2, T&>::type
    operator()(size_t i, size_t j) {
        return data_[i*strides_[0] + j*strides_[1]];
    }
    template<size_t M = N>
    typename std::enable_if<M==2, const T&>::type
    operator()(size_t i, size_t j) const {
        return data_[i*strides_[0] + j*strides_[1]];
    }

    // fill all elements
    void fill(const T& v) {
        std::fill(data_.begin(), data_.end(), v);
    }

    // shape and size
    std::array<size_t, N> shape() const { return shape_; }
    size_t size() const { return data_.size(); }

    // data pointer and iterators
    T*       data()       { return data_.data(); }
    const T* data() const { return data_.data(); }
    auto begin()       { return data_.begin(); }
    auto end()         { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end()   const { return data_.end(); }

    // arithmetic operators
    Tensor operator+(const Tensor& o) const {
        assert(shape_ == o.shape_);
        Tensor r(shape_);
        for (size_t i = 0; i < size(); ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }
    Tensor operator-(const Tensor& o) const {
        assert(shape_ == o.shape_);
        Tensor r(shape_);
        for (size_t i = 0; i < size(); ++i) r.data_[i] = data_[i] - o.data_[i];
        return r;
    }
    Tensor operator*(const Tensor& o) const {
        assert(shape_ == o.shape_);
        Tensor r(shape_);
        for (size_t i = 0; i < size(); ++i) r.data_[i] = data_[i] * o.data_[i];
        return r;
    }
    Tensor operator/(T scalar) const {
        Tensor r(shape_);
        for (size_t i = 0; i < size(); ++i) r.data_[i] = data_[i] / scalar;
        return r;
    }

    // 2D matrix multiplication
    template<size_t M = N>
    typename std::enable_if<M==2, Tensor>::type
    matmul(const Tensor& o) const {
        assert(shape_[1] == o.shape_[0]);
        Tensor r({shape_[0], o.shape_[1]});
        for (size_t i = 0; i < shape_[0]; ++i)
            for (size_t j = 0; j < o.shape_[1]; ++j) {
                T sum{};
                for (size_t k = 0; k < shape_[1]; ++k)
                    sum += (*this)(i,k) * o(k,j);
                r(i,j) = sum;
            }
        return r;
    }
    template<size_t M = N>
    typename std::enable_if<M==2, Tensor>::type
    transpose() const {
        Tensor r({shape_[1], shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i)
            for (size_t j = 0; j < shape_[1]; ++j)
                r(j,i) = (*this)(i,j);
        return r;
    }
};

// stream output
template<typename T, size_t N>
std::ostream& operator<<(std::ostream& os, const Tensor<T,N>& t) {
    if (N == 2) {
        auto s = t.shape(); os << "{\n";
        for (size_t i = 0; i < s[0]; ++i) {
            for (size_t j = 0; j < s[1]; ++j)
                os << t(i,j) << (j+1 < s[1] ? ' ' : '\n');
        }
        os << "}\n";
    } else {
        os << '[';
        for (size_t i = 0; i < t.size(); ++i)
            os << t.data()[i] << (i+1 < t.size() ? ", " : "");
        os << ']';
    }
    return os;
}

} // namespace algebra
} // namespace utec

using utec::algebra::Tensor;

namespace std {

// helper to unpack shape into f(dim0, dim1, ...)
template<typename F, size_t N, size_t... I>
auto apply_impl(F&& f, const array<size_t, N>& arr, index_sequence<I...>)
    -> decltype(std::forward<F>(f)(arr[I]...)) {
    return std::forward<F>(f)(arr[I]...);
}

// element-wise apply: f(element)
template<typename F, typename T, size_t N>
auto apply(const Tensor<T,N>& t, F f)
    -> Tensor<typename result_of<F(T)>::type, N> {
    using U = typename result_of<F(T)>::type;
    Tensor<U, N> r(t.shape());
    const T* in  = t.data();
    U*       out = r.data();
    for (size_t i = 0; i < t.size(); ++i)
        out[i] = f(in[i]);
    return r;
}

// shape unpack apply: f(dim0, dim1, ...)
template<typename F, typename T, size_t N>
auto apply(const Tensor<T,N>& t, F f)
    -> decltype(apply_impl(f, t.shape(), make_index_sequence<N>{})) {
    return apply_impl(f, t.shape(), make_index_sequence<N>{});
}

// tuple-like interface for shape()
template<typename T, size_t N>
struct tuple_size<Tensor<T,N>> : integral_constant<size_t, N> {};

template<size_t I, typename T, size_t N>
struct tuple_element<I, Tensor<T,N>> {
    static_assert(I < N, "Index out of bounds");
    using type = size_t;
};

// get<I>(tensor) → tensor.shape()[I]
template<size_t I, typename T, size_t N>
constexpr auto get(const Tensor<T,N>& t) noexcept -> size_t {
    return t.shape()[I];
}

} // namespace std

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
