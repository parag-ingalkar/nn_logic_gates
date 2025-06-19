#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "../include/tensor.hpp"
#include "../include/linalg.hpp"
#include "../include/activations.hpp"
#include "../include/initializers.hpp"
#include "../include/generate_data.hpp"

void save_model(const Tensor<double> &W1, const Tensor<double> &B1,
                const Tensor<double> &W2, const Tensor<double> &B2,
                const std::string &filename)
{
    std::ofstream file(filename);
    for (auto val : W1.data())
        file << val << " ";
    file << "\n";
    for (auto val : B1.data())
        file << val << " ";
    file << "\n";
    for (auto val : W2.data())
        file << val << " ";
    file << "\n";
    for (auto val : B2.data())
        file << val << " ";
    file << "\n";
    file.close();
}

int main(int argc, char *argv[])
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    // unsigned int seed = 1750363577;
    std::srand(seed);
    std::cout << "Random seed: " << seed << std::endl;

    std::vector<Tensor<double>> X_train;
    std::vector<Tensor<double>> Y_train;

    std::string gate = argv[1];

    if (gate == "xor")
    {
        std::cout << "Generating XOR data..." << std::endl;
        generate_XOR_data(X_train, Y_train);
    }
    else if (gate == "xnor")
    {
        std::cout << "Generating XNOR data..." << std::endl;
        generate_XNOR_data(X_train, Y_train);
    }
    else
    {
        std::cerr << "Unknown gate type: " << gate << ". Use 'xor' or 'xnor'." << std::endl;
        return 1;
    }

    size_t input_size = 2;
    size_t hidden_size = 2; // Number of neurons in the hidden layer
    double learning_rate = 0.1;
    int epochs = 10000; // Number of training epochs

    Tensor<double> W1({input_size, hidden_size});
    Tensor<double> B1({1, hidden_size}, 0.0);
    Tensor<double> W2({hidden_size, 1});
    Tensor<double> B2({1, 1}, 0.0);

    xavier_initialization(W1.data(), 2, 2);
    // randomize(B1.data(), B1.numElements(), -1.0, 1.0);
    xavier_initialization(W2.data(), 2, 1);
    // randomize(B2.data(), B2.numElements(), -1.0, 1.0);

    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double loss = 0.0;

        // Iterate over each training sample
        for (size_t i = 0; i < X_train.size(); ++i)
        {
            Tensor<double> x = X_train[i];
            Tensor<double> y = Y_train[i];

            // Forward pass
            Tensor<double> z1 = dot(x, W1) + B1;
            Tensor<double> a1 = tanh(z1);
            Tensor<double> z2 = dot(a1, W2) + B2;
            Tensor<double> a2 = sigmoid(z2);

            double y_true = y.data()[0];
            double y_pred = a2.data()[0];
            loss += -(y_true * std::log(y_pred + 1e-8) + (1 - y_true) * std::log(1 - y_pred + 1e-8));

            // Backward pass
            Tensor<double> dZ2 = a2 - y;
            Tensor<double> dW2 = dot(transpose(a1), dZ2);
            Tensor<double> dB2 = dZ2;

            Tensor<double> dA1 = dot(dZ2, transpose(W2));
            Tensor<double> dZ1 = grad_tanh(a1);

            for (size_t j = 0; j < dZ1.numElements(); ++j)
            {
                dZ1.data()[j] *= dA1.data()[j]; // Chain rule
            }
            Tensor<double> dW1 = dot(transpose(x), dZ1);
            Tensor<double> dB1 = dZ1;

            // Update weights and biases
            W2 = W2 - (learning_rate * dW2);
            B2 = B2 - (learning_rate * dB2);
            W1 = W1 - (learning_rate * dW1);
            B1 = B1 - (learning_rate * dB1);
        }

        loss /= X_train.size(); // Average loss for the epoch
        if (epoch % 1000 == 0)
        {
            std::cout << "Epoch " << epoch << " Loss: " << loss << std::endl;
        }
    }

    // Testing the trained model
    std::cout << "Testing the trained model:" << std::endl;
    for (size_t i = 0; i < X_train.size(); ++i)
    {
        Tensor<double> x = X_train[i];
        Tensor<double> z1 = dot(x, W1) + B1;
        Tensor<double> a1 = tanh(z1);
        Tensor<double> z2 = dot(a1, W2) + B2;
        Tensor<double> a2 = sigmoid(z2);
        std::cout << "Input: " << x << "Expected: " << Y_train[i] << "Predicted: " << a2 << std::endl;
    }

    // Save the model
    std::string filename = "../models/" + gate + "_model.txt";
    save_model(W1, B1, W2, B2, filename);
    std::cout << "Saved the model to " << filename << std::endl;

    return 0;
}