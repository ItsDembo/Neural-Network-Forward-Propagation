#!/usr/bin/env python3
"""
Visualization example for the neural network, showing layer connections.
Requires matplotlib for visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from nn_forward_prop import NeuralNetwork

def draw_neural_net(ax, layer_sizes):
    """
    Draw a neural network diagram using matplotlib.
    
    Args:
        ax: matplotlib axes
        layer_sizes: list of layer sizes
    """
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                              v_spacing/4.,
                              color='w', ec='k', zorder=4)
            ax.add_patch(circle)
    
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                 [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], 
                                 c='k')
                ax.add_patch(line)

def main():
    # Create a sample network
    network = NeuralNetwork([3, 4, 2])
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 6))
    ax = fig.gca()
    ax.axis('off')
    
    # Draw the network
    draw_neural_net(ax, network.layers)
    
    # Add title and information
    plt.title(f"Neural Network Architecture: {network.layers}", y=1.1)
    plt.text(0.5, -0.1, 
             f"Input Layer: {network.layers[0]} neurons\n"
             f"Hidden Layer: {network.layers[1]} neurons\n"
             f"Output Layer: {network.layers[2]} neurons",
             ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('network_architecture.png')
    print("Network visualization saved as 'network_architecture.png'")
    plt.show()

if __name__ == "__main__":
    main()