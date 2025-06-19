#include <iostream>
#include "tensor.hpp"
#include "linalg.hpp"

int main()
{
    // Create a 2D tensor with shape (2, 3)
    Tensor<int> A({1, 3});
    A({0, 0}) = 2;
    A({0, 1}) = 3;
    A({0, 2}) = 4;

    std::cout << "A = \n"
              << A << std::endl;

    Tensor<int> B({3, 2});
    B({0, 0}) = 0;
    B({0, 1}) = 1000;
    B({1, 0}) = 1;
    B({1, 1}) = 100;
    B({2, 0}) = 0;
    B({2, 1}) = 10;

    std::cout << "B = \n"
              << B << std::endl;

    Tensor<int> C = dot(A, B);
    std::cout << "A . B = \n"
              << C << std::endl;

    Tensor<int> D = transpose(A);
    std::cout << "A^T = \n"
              << D << std::endl;

    Tensor<int> E = 2 * A;
    std::cout << "2 * A = \n"
              << E << std::endl;

    Tensor<int> F = A + A;
    std::cout << "A + A = \n"
              << F << std::endl;

    Tensor<int> G = B - B;
    std::cout << "B - B = \n"
              << G << std::endl;

    Tensor<int> H = dot(transpose(B), transpose(A));
    std::cout << "B^T . A^T = \n"
              << H << std::endl;
    return 0;
}