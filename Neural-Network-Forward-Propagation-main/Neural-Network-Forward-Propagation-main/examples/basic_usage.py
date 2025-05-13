#!/usr/bin/env python3

"""
Basic usage example for the neural network forward propagation implementation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_forward_prop import NeuralNetwork


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
    
    # Print results
    print("\nBasic Neural Network Example")
    print("="*40)
    print(f"Network architecture: {network.layers}")
    print(f"Input values: {inputs}\n")
    
    for i, output in enumerate(hidden_outputs):
        print(f"Hidden layer {i+1} outputs: {[round(x, 4) for x in output]}")
    
    print(f"\nFinal output predictions: {[round(x, 4) for x in predictions]}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()