#include "tensor.hpp"

// Scalar multiplication of a tensor
template <typename A, typename T>
Tensor<T> operator*(const A &scalar, const Tensor<T> &tensor)
{
    Tensor<T> result(tensor.shape());
    for (size_t i = 0; i < tensor.numElements(); ++i)
    {
        result.data()[i] = scalar * tensor.data()[i];
    }
    return result;
}

// Addition of two tensors
template <typename T>
Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b)
{
    assert(a.shape() == b.shape());
    Tensor<T> result(a.shape());
    for (size_t i = 0; i < a.numElements(); ++i)
    {
        result.data()[i] = a.data()[i] + b.data()[i];
    }
    return result;
}

// Subtraction of two tensors
template <typename T>
Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b)
{
    assert(a.shape() == b.shape());
    Tensor<T> result(a.shape());
    result = a + (-1 * b);
    return result;
}

// Transpose of a tensor
template <typename T>
Tensor<T> transpose(const Tensor<T> &tensor)
{
    assert(tensor.rank() == 2);
    Tensor<T> result({tensor.shape()[1], tensor.shape()[0]});
    for (size_t i = 0; i < tensor.shape()[0]; ++i)
    {
        for (size_t j = 0; j < tensor.shape()[1]; ++j)
        {
            result({j, i}) = tensor({i, j});
        }
    }
    return result;
}

// Dot product of two tensors
template <typename T>
Tensor<T> dot(const Tensor<T> &a, const Tensor<T> &b)
{
    assert(a.shape()[1] == b.shape()[0]);
    Tensor<T> result({a.shape()[0], b.shape()[1]});
    for (size_t i = 0; i < a.shape()[0]; ++i)
    {
        for (size_t j = 0; j < b.shape()[1]; ++j)
        {
            T sum = 0;
            for (size_t k = 0; k < a.shape()[1]; ++k)
            {
                sum += a({i, k}) * b({k, j});
            }
            result({i, j}) = sum;
        }
    }
    return result;
}