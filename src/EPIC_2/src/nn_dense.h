#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H

#pragma once

#include "nn_interfaces.h"
#include "tensor.h"

namespace utec {
namespace neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    utec::algebra::Tensor<T,2> weights_;
    utec::algebra::Tensor<T,2> biases_;
    utec::algebra::Tensor<T,2> last_input_;
    utec::algebra::Tensor<T,2> weight_gradients_;
    utec::algebra::Tensor<T,2> bias_gradients_;

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_features, size_t out_features,
          InitWFun init_w_fun, InitBFun init_b_fun)
      : weights_(in_features, out_features),
        biases_(1, out_features),
        weight_gradients_(in_features, out_features),
        bias_gradients_(1, out_features)
    {
        init_w_fun(weights_);
        init_b_fun(biases_);
        weight_gradients_.fill(static_cast<T>(0));
        bias_gradients_.fill(static_cast<T>(0));
    }

    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& x) override {
        last_input_ = x;
        auto shape = x.shape();
        size_t batch = shape[0], out_f = weights_.shape()[1];
        utec::algebra::Tensor<T,2> out(batch, out_f);
        for (size_t i = 0; i < batch; ++i) {
            for (size_t j = 0; j < out_f; ++j) {
                T sum = 0;
                for (size_t k = 0; k < weights_.shape()[0]; ++k)
                    sum += x(i,k) * weights_(k,j);
                out(i,j) = sum + biases_(0,j);
            }
        }
        return out;
    }

    utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& grad) override {
        auto gshape = grad.shape();
        size_t batch = gshape[0],
               out_f = gshape[1],
               in_f  = last_input_.shape()[1];

        // Gradientes de pesos
        weight_gradients_.fill(0);
        for (size_t i = 0; i < in_f; ++i)
          for (size_t j = 0; j < out_f; ++j)
            for (size_t k = 0; k < batch; ++k)
              weight_gradients_(i,j) += last_input_(k,i) * grad(k,j);

        // Gradientes de sesgos
        bias_gradients_.fill(0);
        for (size_t j = 0; j < out_f; ++j)
          for (size_t k = 0; k < batch; ++k)
            bias_gradients_(0,j) += grad(k,j);

        // Propaga hacia atrás al input
        utec::algebra::Tensor<T,2> in_grad(batch, in_f);
        for (size_t i = 0; i < batch; ++i)
          for (size_t j = 0; j < in_f; ++j) {
            T sum = 0;
            for (size_t k = 0; k < out_f; ++k)
              sum += grad(i,k) * weights_(j,k);
            in_grad(i,j) = sum;
          }
        return in_grad;
    }

    void update_params(IOptimizer<T>& opt) override {
        opt.update(weights_, weight_gradients_);
        opt.update(biases_, bias_gradients_);
    }
};

} // namespace neural_network
} // namespace utec

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_DENSE_H
