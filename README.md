# Neural Network Forward Propagation

A clean Python implementation of forward propagation in artificial neural networks.

## Features

- Simple neural network implementation
- Forward propagation through arbitrary network architectures
- Sigmoid activation function
- Easy to understand and extend

## Usage

```python
from nn_forward_prop import NeuralNetwork

# Create a network with 2 inputs, one hidden layer (3 neurons), and 1 output
network = NeuralNetwork([2, 3, 1])

# Forward propagate some inputs
inputs = [0.5, 0.8]
predictions, hidden_outputs = network.forward_propagate(inputs)

print("Network predictions:", predictions)
