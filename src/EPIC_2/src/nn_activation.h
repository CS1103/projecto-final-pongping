
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H

#pragma once

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec {
namespace neural_network {

template<typename T>
class ReLU final : public ILayer<T> {
private:
    utec::algebra::Tensor<T,2> last_input_;

public:
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
        last_input_ = z; // Store input for backward pass
        utec::algebra::Tensor<T,2> result(z.shape());

        const auto& shape = z.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result(i, j) = std::max(static_cast<T>(0), z(i, j));
            }
        }
        return result;
    }

    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& gradient) override {
        utec::algebra::Tensor<T,2> result(gradient.shape());

        const auto& shape = gradient.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result(i, j) = (last_input_(i, j) > 0) ? gradient(i, j) : static_cast<T>(0);
            }
        }
        return result;
    }
};

template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    utec::algebra::Tensor<T,2> last_output_;

public:
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& z) override {
        utec::algebra::Tensor<T,2> result(z.shape());

        const auto& shape = z.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T x = z(i, j);
                x = std::max(static_cast<T>(-500), std::min(static_cast<T>(500), x));
                result(i, j) = static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
            }
        }

        last_output_ = result; // Store output for backward pass
        return result;
    }

    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& gradient) override {
        utec::algebra::Tensor<T,2> result(gradient.shape());

        const auto& shape = gradient.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T sigmoid_val = last_output_(i, j);
                result(i, j) = gradient(i, j) * sigmoid_val * (static_cast<T>(1) - sigmoid_val);
            }
        }
        return result;
    }
};

}
}
#endif // PROG3_NN_FINAL_PROJECT_V2025_01_ACTIVATION_H