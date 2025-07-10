
#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H

#pragma once

#include "tensor.h"

namespace utec {
    namespace neural_network {

        template<typename T>
        class IOptimizer;

        template<typename T>
        class ILayer {
        public:
            virtual ~ILayer() = default;
            virtual utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& input) = 0;
            virtual utec::algebra::Tensor<T,2> backward(const utec::algebra::Tensor<T,2>& gradient) = 0;
            virtual void update_params(IOptimizer<T>& optimizer) {}
        };

        template<typename T, size_t N>
        class ILoss {
        public:
            virtual ~ILoss() = default;
            virtual T loss() const = 0;
            virtual utec::algebra::Tensor<T,N> loss_gradient() const = 0;
        };

        template<typename T>
        class IOptimizer {
        public:
            virtual ~IOptimizer() = default;
            virtual void update(utec::algebra::Tensor<T,2>& params, const utec::algebra::Tensor<T,2>& grads) = 0;
            virtual void step() {}
        };

    }
}
#endif // PROG3_NN_FINAL_PROJECT_V2025_01_LAYER_H
