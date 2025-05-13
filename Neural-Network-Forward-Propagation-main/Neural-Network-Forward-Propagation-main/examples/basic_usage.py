#!/usr/bin/env python3
"""
Basic usage example for the neural network forward propagation implementation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_forward_prop import NeuralNetwork

def format_output(values):
    """Format numpy float values to clean 4-digit floats"""
    return [float(f"{x:.4f}") for x in values]

def main():
    # Create a network with:
    # - 3 input neurons
    # - 2 hidden layers with 4 and 3 neurons respectively
    # - 2 output neurons
    network = NeuralNetwork([3, 4, 3, 2])
    
    # Example input
    inputs = [0.5, 0.3, 0.8]
    
    # Forward propagate
    predictions, hidden_outputs = network.forward_propagate(inputs)
    
    # Print results with clean formatting
    print("\nBasic Neural Network Example")
    print("="*40)
    print(f"Network architecture: {network.layers}")
    print(f"Input values: {inputs}\n")
    
    for i, output in enumerate(hidden_outputs):
        print(f"Hidden layer {i+1} outputs: {format_output(output)}")
    
    print(f"\nFinal output predictions: {format_output(predictions)}")
    print("="*40)
    
    # Print weights and biases for reference
    print("\nNetwork Parameters:")
    for i, (weight, bias) in enumerate(zip(network.weights, network.biases)):
        print(f"\nLayer {i+1} to {i+2}:")
        print(f"Weight matrix ({weight.shape[0]}x{weight.shape[1]}):")
        print(weight.round(4))
        print(f"Bias vector: {bias.round(4)}")

if __name__ == "__main__":
    main()
