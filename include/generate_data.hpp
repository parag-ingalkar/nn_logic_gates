#ifndef GENERATE_DATA_HPP
#define GENERATE_DATA_HPP

#include <vector>
#include "tensor.hpp"

// Generates XOR training data
void generate_XOR_data(std::vector<Tensor<double>> &X_train, std::vector<Tensor<double>> &Y_train)
{
    X_train.clear();
    Y_train.clear();

    std::vector<std::pair<int, int>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};

    for (const auto &input : inputs)
    {
        Tensor<double> x({1, 2});
        x({0, 0}) = input.first;
        x({0, 1}) = input.second;
        X_train.push_back(x);

        Tensor<double> y({1, 1});
        y({0, 0}) = input.first ^ input.second; // XOR operation
        Y_train.push_back(y);
    }
}

// Generates XNOR training data
void generate_XNOR_data(std::vector<Tensor<double>> &X_train, std::vector<Tensor<double>> &Y_train)
{
    X_train.clear();
    Y_train.clear();
    std::vector<std::pair<int, int>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}};
    for (const auto &input : inputs)
    {
        Tensor<double> x({1, 2});
        x({0, 0}) = input.first;
        x({0, 1}) = input.second;
        X_train.push_back(x);

        Tensor<double> y({1, 1});
        y({0, 0}) = !(input.first ^ input.second); // XNOR operation
        Y_train.push_back(y);
    }
}

#endif