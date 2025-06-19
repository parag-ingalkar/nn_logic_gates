cmake_minimum_required(VERSION 3.10)
project(NN_from_scratch)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include headers
include_directories(include)

# Add executables
add_executable(train src/train_logic_nn.cpp)
add_executable(test src/test.cpp)
