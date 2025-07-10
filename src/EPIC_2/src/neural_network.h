#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H

#pragma once

#include "tensor.h"
#include "nn_interfaces.h"
#include "nn_optimizer.h"
#include "nn_loss.h"

#include <vector>
#include <memory>

namespace utec {
namespace neural_network {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::shared_ptr<ILayer<T>>> layers_;

public:
    NeuralNetwork() = default;

    // Para los tests que llaman a net.add_layer(...)
    void add_layer(std::shared_ptr<ILayer<T>> layer) {
        layers_.push_back(layer);
    }

    // Alias opcional
    void add(std::shared_ptr<ILayer<T>> layer) {
        add_layer(layer);
    }

    // Propagación forward
    utec::algebra::Tensor<T,2> forward(const utec::algebra::Tensor<T,2>& input) {
        utec::algebra::Tensor<T,2> x = input;
        for (auto& l : layers_) {
            x = l->forward(x);
        }
        return x;
    }

    // Alias que usan los tests: net.predict(...)
    utec::algebra::Tensor<T,2> predict(const utec::algebra::Tensor<T,2>& input) {
        return forward(input);
    }

    // Entrenamiento: por defecto usa SGD<T> como optimizador
    template<template<class...> class LossType,
             template<class...> class OptimizerType = SGD>
    void train(const utec::algebra::Tensor<T,2>& X,
               const utec::algebra::Tensor<T,2>& Y,
               size_t epochs, size_t batch_size, T lr)
    {
        OptimizerType<T> opt(lr);

        for (size_t e = 0; e < epochs; ++e) {
            // forward
            auto y_pred = forward(X);

            // calcula pérdida y gradiente
            LossType<T> loss(y_pred, Y);
            auto grad = loss.loss_gradient();

            // backprop
            for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
                grad = (*it)->backward(grad);
            }

            // actualiza parámetros
            for (auto& l : layers_) {
                l->update_params(opt);
            }
        }
    }
};

} // namespace neural_network
} // namespace utec

#endif // PROG3_NN_FINAL_PROJECT_V2025_01_NEURAL_NETWORK_H
