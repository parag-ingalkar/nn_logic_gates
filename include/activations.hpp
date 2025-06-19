#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "tensor.hpp"
#include <cmath>

Tensor<double> relu(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        result.data()[i] = std::max(0.0, x.data()[i]);
    }
    return result;
}

Tensor<double> grad_relu(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        result.data()[i] = x.data()[i] > 0 ? 1.0 : 0.0;
    }
    return result;
}

Tensor<double> sigmoid(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        result.data()[i] = 1.0 / (1.0 + std::exp(-x.data()[i]));
    }
    return result;
}

Tensor<double> grad_sigmoid(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        result.data()[i] = x.data()[i] * (1.0 - x.data()[i]);
    }
    return result;
}

Tensor<double> tanh(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        result.data()[i] = std::tanh(x.data()[i]);
    }
    return result;
}

Tensor<double> grad_tanh(const Tensor<double> &x)
{
    Tensor<double> result(x.shape());
    for (size_t i = 0; i < x.numElements(); ++i)
    {
        double t = std::tanh(x.data()[i]);
        result.data()[i] = 1.0 - t * t;
    }
    return result;
}

#endif // ACTIVATIONS_HPP