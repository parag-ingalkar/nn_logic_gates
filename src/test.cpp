#include "../include/tensor.hpp"
#include "../include/linalg.hpp"
#include "../include/activations.hpp"

#include <iostream>

void load_model(Tensor<double> &W1, Tensor<double> &B1,
                Tensor<double> &W2, Tensor<double> &B2,
                const std::string &filename)
{
    std::ifstream file(filename);
    for (auto &val : W1.data())
        file >> val;
    for (auto &val : B1.data())
        file >> val;
    for (auto &val : W2.data())
        file >> val;
    for (auto &val : B2.data())
        file >> val;
    file.close();
}

int main()
{
    size_t input_size = 2, hidden_size = 2;
    Tensor<double> W1({input_size, hidden_size});
    Tensor<double> B1({1, hidden_size});
    Tensor<double> W2({hidden_size, 1});
    Tensor<double> B2({1, 1});

    std::vector<std::string> gate_names = {"xor", "xnor"};

    while (true)
    {
        std::cout << "\nChoose a logic gate:\n";
        for (size_t i = 0; i < gate_names.size(); ++i)
            std::cout << i + 1 << ". " << gate_names[i] << "\n";
        std::cout << gate_names.size() + 1 << ". Exit\n";

        int choice;
        std::cout << "Enter choice: ";
        std::cin >> choice;

        if (choice == gate_names.size() + 1)
            break;

        if (choice < 1 || choice > gate_names.size())
        {
            std::cout << "Invalid choice.\n";
            continue;
        }

        std::string selected_gate = gate_names[choice - 1];
        std::cout << "Loading model for " << selected_gate << " gate...\n";
        std::string filename = "../models/" + selected_gate + "_model.txt";
        load_model(W1, B1, W2, B2, filename);

        while (true)
        {
            double a, b;
            std::cout << "\nEnter input (0 or 1) for a and b (or -1 to choose another gate): ";
            std::cin >> a;

            if (a == -1)
                break;

            std::cin >> b;
            if (b == -1)
                break;

            Tensor<double> x({1, 2});
            x({0, 0}) = a;
            x({0, 1}) = b;

            Tensor<double> z1 = dot(x, W1) + B1;
            Tensor<double> a1 = tanh(z1);
            Tensor<double> z2 = dot(a1, W2) + B2;
            Tensor<double> output = sigmoid(z2);

            std::cout << "Predicted: " << output.data()[0]
                      << std::endl;
        }
    }

    return 0;
}
