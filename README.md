# Implementing a Multi-Layer Perceptron (MLP) in C++

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CMake](https://img.shields.io/badge/CMake-3.17+-green.svg)](https://cmake.org/)

A comprehensive implementation of a Multi-Layer Perceptron (MLP) neural network from scratch using pure C++. This project leverages the powerful **microgradpp** library to provide a complete neural network framework with automatic differentiation, memory management, and modern C++ design patterns.

##  Features

- **Pure C++ Implementation**: High-performance neural networks built from the ground up
- **Automatic Differentiation**: Built-in backpropagation through computational graphs
- **Memory Management**: Cross-platform memory usage tracking and optimization
- **Modern C++17**: Leverages modern C++ features for clean, efficient code
- **Header-Only Library**: Flexible usage as header-only or compiled library
- **Cross-Platform**: Supports macOS, Linux, and Windows
- **Comprehensive Examples**: MLP training, computer vision, and memory management demos
- **Extensible Architecture**: Easy to extend with custom layers and activation functions

## 📁 Project Structure

```
Implementing-a-Multi-Layer-Perceptron-MLP-in-C-/
├── external/
│   └── microgradpp/          # Core neural network library (Git submodule)
│       ├── include/          # Header files
│       ├── examples/         # Usage examples
│       ├── tests/           # Test suite
│       └── CMakeLists.txt   # Build configuration
├── .github/
│   └── workflow/            # GitHub Actions for automated updates
├── LICENSE                  # GNU GPL v3 License
└── README.md               # This file
```

## Prerequisites

- **CMake**: Version 3.17 or higher
- **C++ Compiler**: Supporting C++17 standard (GCC 7+, Clang 5+, MSVC 2017+)
- **OpenCV**: For visualization features (optional)
- **TBB**: Intel Threading Building Blocks for parallel processing

## Building the Project

### 1. Clone the Repository
```bash
git clone https://github.com/eltonbaidoo/Implementing-a-Multi-Layer-Perceptron-MLP-in-C-.git
cd Implementing-a-Multi-Layer-Perceptron-MLP-in-C-
git submodule update --init --recursive
```

### 2. Build Options

#### Basic Build (Header-only library)
```bash
cd external/microgradpp
mkdir build && cd build
cmake ..
make
```

#### Build with Examples and Tests
```bash
cd external/microgradpp
mkdir build && cd build
cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=ON ..
make
```

#### Release Build
```bash
cd external/microgradpp
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

##  Quick Start

### Basic Usage Example

```cpp
#include "external/microgradpp/include/Value.hpp"
#include "external/microgradpp/include/Neuron.hpp"
#include "external/microgradpp/include/Tensor.hpp"

int main() {
    // Create a simple neural network
    auto a = microgradpp::Value::create(2.0);
    auto b = microgradpp::Value::create(3.0);
    auto c = a * b;
    
    // Automatic differentiation
    c->backProp();
    
    std::cout << "Gradient of a: " << a->grad << std::endl;
    std::cout << "Gradient of b: " << b->grad << std::endl;
    
    return 0;
}
```

### Multi-Layer Perceptron Example

```cpp
#include "external/microgradpp/include/base/BaseMultiLayerPerceptron.hpp"
#include "external/microgradpp/include/core/Sequential.hpp"

class MyMLP : public microgradpp::base::BaseMultiLayerPerceptron {
public:
    MyMLP() : BaseMultiLayerPerceptron(
        microgradpp::core::Sequential({
            microgradpp::nn::Linear(10, 8),    // Input: 10, Hidden: 8
            microgradpp::nn::ReLU(),           // Activation
            microgradpp::nn::Linear(8, 1)      // Output: 1
        })
    ) {
        this->learningRate = 0.001;
    }
    
    microgradpp::Tensor1D forward(microgradpp::Tensor1D input) override {
        return this->sequential(input);
    }
};
```

##  Core Components

### 1. **Value Class**
The fundamental building block for automatic differentiation:
- Stores numerical values and gradients
- Maintains computational graph connections
- Supports mathematical operations with automatic gradient computation

### 2. **Neuron Class**
Represents individual neurons in the network:
- Weights and bias management
- Activation function support
- Gradient tracking and updates

### 3. **Tensor Classes**
Multi-dimensional data structures:
- `Tensor1D`: 1-dimensional tensors
- `Tensor2D`: 2-dimensional tensors
- Automatic gradient management
- Efficient memory operations

### 4. **Sequential Architecture**
Layer stacking and forward propagation:
- Chain multiple layers together
- Automatic forward pass through all layers
- Unified parameter management

##  Examples

### 1. **MLP Training**
Run the multi-layer perceptron example:
```bash
cd external/microgradpp/build/examples
./mlp
```

### 2. **Computer Vision**
Learn to predict German Shepherd faces:
```bash
cd external/microgradpp/build/examples
./images
```

### 3. **Memory Management**
Monitor memory usage during training:
```bash
cd external/microgradpp/build
./m++
```

##  Testing

Run the comprehensive test suite:
```bash
cd external/microgradpp/build/tests
make
./run_tests
```

##  Automated Updates

This project includes GitHub Actions that automatically update the microgradpp submodule weekly to ensure you have the latest features and bug fixes.

##  Performance Features

- **Memory Tracking**: Cross-platform memory usage monitoring
- **Gradient Clipping**: Prevents exploding gradients
- **Optimized Operations**: Efficient tensor operations
- **Parallel Processing**: TBB integration for multi-threading

##  Contributing

We welcome contributions! Here are some areas where you can help:

- **Bug Fixes**: Report and fix issues
- **Documentation**: Improve examples and documentation
- **New Features**: Add new activation functions, layers, or optimizers
- **Performance**: Optimize existing implementations
- **Testing**: Add more comprehensive tests

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Andrej Karpathy**: For the original [micrograd](https://github.com/karpathy/micrograd) library
- **Gautam Sharma**: Creator of the [microgradpp](https://github.com/ggsharma/microgradpp) library
- **Open Source Community**: For various tools and libraries used in this project

##  Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Discussions**: Join discussions in GitHub Discussions
- **Documentation**: Check the [microgradpp documentation](https://github.com/ggsharma/microgradpp)

---

**Built with using modern C++ and the power of automatic differentiation**
