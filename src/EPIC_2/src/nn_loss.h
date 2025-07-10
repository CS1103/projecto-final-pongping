
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H

#pragma once

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec {
namespace neural_network {

template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T,2> y_prediction_;
    utec::algebra::Tensor<T,2> y_true_;

public:
    MSELoss(const utec::algebra::Tensor<T,2>& y_prediction, const utec::algebra::Tensor<T,2>& y_true)
        : y_prediction_(y_prediction), y_true_(y_true) {
        assert(y_prediction.shape() == y_true.shape());
    }

    T loss() const override {
        T total_loss = 0;
        const auto& shape = y_prediction_.shape();
        size_t total_elements = y_prediction_.size();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T diff = y_prediction_(i, j) - y_true_(i, j);
                total_loss += diff * diff;
            }
        }

        return total_loss / static_cast<T>(total_elements);
    }

    utec::algebra::Tensor<T,2> loss_gradient() const override {
        utec::algebra::Tensor<T,2> gradient(y_prediction_.shape());
        const auto& shape = y_prediction_.shape();
        T factor = static_cast<T>(2) / static_cast<T>(y_prediction_.size());
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                gradient(i, j) = factor * (y_prediction_(i, j) - y_true_(i, j));
            }
        }
        return gradient;
    }
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    utec::algebra::Tensor<T,2> y_prediction_;
    utec::algebra::Tensor<T,2> y_true_;

public:
    BCELoss(const utec::algebra::Tensor<T,2>& y_prediction, const utec::algebra::Tensor<T,2>& y_true)
        : y_prediction_(y_prediction), y_true_(y_true) {
        assert(y_prediction.shape() == y_true.shape());
    }

    T loss() const override {
        T total_loss = 0;
        const auto& shape = y_prediction_.shape();
        size_t total_elements = y_prediction_.size();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T yp = std::max(std::min(y_prediction_(i, j), static_cast<T>(1) - 1e-7), static_cast<T>(1e-7));
                T yt = y_true_(i, j);
                total_loss += -(yt * std::log(yp) + (static_cast<T>(1) - yt) * std::log(static_cast<T>(1) - yp));
            }
        }
        return total_loss / static_cast<T>(total_elements);
    }

    utec::algebra::Tensor<T,2> loss_gradient() const override {
        utec::algebra::Tensor<T,2> gradient(y_prediction_.shape());
        const auto& shape = y_prediction_.shape();
        size_t total_elements = y_prediction_.size();

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T yp = std::max(std::min(y_prediction_(i, j), static_cast<T>(1) - 1e-7), static_cast<T>(1e-7));
                T yt = y_true_(i, j);
                gradient(i, j) = (yp - yt) / ((yp) * (1 - yp) * static_cast<T>(total_elements));
            }
        }
        return gradient;
    }
};

}
}
#endif // PROG3_NN_FINAL_PROJECT_V2025_01_LOSS_H
