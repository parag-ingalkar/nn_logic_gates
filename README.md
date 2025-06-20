# NN_from_scratch

This project implements simple neural networks for logic gates in C++ with custom tensor and linear algebra code.

## Project structure

```
NN_from_scratch
├──include
│   ├──activations.hpp                  # Activation functions
│   ├──generate_data.hpp                # Generate LOGIC GATE data
│   ├──initializers.hpp                 # Initializers for Weight initialization
│   ├──linalg.hpp                       # Linear Algebra for Tensor Class
│   └──tensor.hpp                       # Custom Tensor Class
├──models
│   ├──xnor_model.txt                   # Pretrained Model for XNOR
│   └──xor_model.txt                    # Pretrained Model for XOR
├──src
│   ├──test_tensor_calculations.cpp     # Various Linear Algebra operations for Tensors
│   ├──test.cpp                         # Test your Models
│   └──train_logic_nn.cpp               # Train your Models
├──CMakeLists.txt
└──README.md
```

## Build & Run

```bash
mkdir build && cd build
cmake ..
make
./train xor                 # To train xor -> Saves model to models/xor_model.txt
./train xnor                 # To train xnor -> Saves model to models/xnor_model.txt
./test
```

Once you run the ./test you will be prompted as
(Example to select XOR Gate)

```bash
Choose a logic gate:
1. xor
2. xnor
3. Exit
Enter choice: 1
```

Enter your choice to load the respective model from models/

Upon Selection, the respective model will load and you will be prompted to enter the input values
(Example input as 0 1)

```bash
Enter input (0 or 1) for a and b (or -1 to choose another gate): 0 1
```

You should see the predicted value in the output.
(Example output)

```bash
Predicted: 0.998475
```

## Features

- tensor.hpp contains implementaion of custom Tensor class with basic required functionality.
- linalg.hpp contains implementaion of basic Tensor (Matrix-Vector) Linear Algebra necessary for Neural Networks
- The Implementation provides funtionality to Save and Load your trained model.
- The save_model() function saves the parameters in a .txt file.
- The load_model() function loads the parameters from the .txt file.
