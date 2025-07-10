#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H

#pragma once

#include "nn_interfaces.h"
#include "tensor.h"
#include <cmath>

namespace utec {
namespace neural_network {

// **SGD** (ya lo tenías)
template<typename T>
class SGD final : public IOptimizer<T> {
    T lr_;
public:
    explicit SGD(T learning_rate) : lr_(learning_rate) {}
    void update(utec::algebra::Tensor<T,2>& P,
                const utec::algebra::Tensor<T,2>& G) override {
        auto s = P.shape();
        for (size_t i = 0; i < s[0]; ++i)
            for (size_t j = 0; j < s[1]; ++j)
                P(i,j) -= lr_ * G(i,j);
    }
};

// **Adam** con las tres firmas que prueban tus tests
template<typename T>
class Adam final : public IOptimizer<T> {
    T lr_, beta1_, beta2_, eps_;
    utec::algebra::Tensor<T,2> m_w_, v_w_, m_b_, v_b_;
    size_t t_ = 0;

public:
    // sólo lr
    explicit Adam(T learning_rate)
      : Adam(learning_rate, T(0.9), T(0.999), T(1e-8)) {}

    // lr, beta1, beta2
    Adam(T learning_rate, T beta1, T beta2)
      : Adam(learning_rate, beta1, beta2, T(1e-8)) {}

    // lr, beta1, beta2, eps
    Adam(T learning_rate, T beta1, T beta2, T epsilon)
      : lr_(learning_rate), beta1_(beta1), beta2_(beta2), eps_(epsilon)
    {}

    void update(utec::algebra::Tensor<T,2>& P,
                const utec::algebra::Tensor<T,2>& G) override {
        auto s = P.shape();
        // inicializar m, v la primera vez
        if (t_ == 0) {
            m_w_ = utec::algebra::Tensor<T,2>(s);
            v_w_ = utec::algebra::Tensor<T,2>(s);
            m_b_ = m_w_;  // forma {1, out_features}
            v_b_ = v_w_;
        }
        ++t_;

        // update moment estimates y parámetros
        for (size_t i = 0; i < s[0]; ++i) {
            for (size_t j = 0; j < s[1]; ++j) {
                T g = G(i,j);
                // actualizar m, v
                m_w_(i,j) = beta1_*m_w_(i,j) + (T(1)-beta1_)*g;
                v_w_(i,j) = beta2_*v_w_(i,j) + (T(1)-beta2_)*(g*g);
                // bias correction
                T m_hat = m_w_(i,j) / (T(1) - std::pow(beta1_, t_));
                T v_hat = v_w_(i,j) / (T(1) - std::pow(beta2_, t_));
                // paso de Adam
                P(i,j) -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

} // namespace neural_network
} // namespace utec

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_OPTIMIZER_H
