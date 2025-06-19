#ifndef _INITIALIZERS_HPP
#define _INITIALIZERS_HPP

#include <vector>
#include <cmath>

double rand_uniform(double min, double max)
{
    return min + (double(rand()) / RAND_MAX) * (max - min);
}

void random_initialization(std::vector<double> &data, double min, double max)
{

    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = rand_uniform(min, max);
    }
}

void xavier_initialization(std::vector<double> &data, size_t fan_in, size_t fan_out)
{
    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = rand_uniform(-limit, limit);
    }
}

#endif