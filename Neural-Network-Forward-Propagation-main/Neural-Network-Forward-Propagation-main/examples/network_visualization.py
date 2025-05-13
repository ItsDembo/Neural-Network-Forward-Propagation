#!/usr/bin/env python3
"""
Enhanced neural network visualization with separated weight and bias labels.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nn_forward_prop import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

def draw_neural_net(ax, network, input_values):
    """
    Draw neural network with weights on connections and biases above nodes.
    """
    layer_sizes = network.layers
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    # Forward pass to get activations
    activations = [np.array(input_values)]
    for i in range(len(network.weights)):
        layer_input = activations[-1]
        layer_output = network.sigmoid(np.dot(network.weights[i], layer_input) + network.biases[i])
        activations.append(layer_output)
    
    # Draw nodes with activations
    for n, (layer_size, layer_activations) in enumerate(zip(layer_sizes, activations)):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            # Node circle
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), 
                              v_spacing/4.,
                              color='w', ec='k', zorder=4)
            ax.add_patch(circle)
            
            # Activation label inside node
            ax.text(n*h_spacing + left, 
                   layer_top - m*v_spacing, 
                   f"{layer_activations[m]:.2f}",
                   ha='center', va='center', fontsize=8)
            
            # Bias label above node (except input layer)
            if n > 0:
                ax.text(n*h_spacing + left,
                       layer_top - m*v_spacing + v_spacing/3,
                       f"b={network.biases[n-1][m]:.2f}",
                       fontsize=7, ha='center', color='blue')
    
    # Draw connections with weights
    for i in range(len(network.weights)):
        for j in range(network.weights[i].shape[0]):
            for k in range(network.weights[i].shape[1]):
                x1 = i*h_spacing + left
                y1 = v_spacing*(network.layers[i] - 1)/2. + (top + bottom)/2. - k*v_spacing
                x2 = (i+1)*h_spacing + left
                y2 = v_spacing*(network.layers[i+1] - 1)/2. + (top + bottom)/2. - j*v_spacing
                
                # Connection line
                line = ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3)[0]
                
                # Weight label - placed closer to destination node
                label_pos = 0.7  # 0.5=midpoint, 1.0=destination
                label_x = x1 + (x2-x1)*label_pos
                label_y = y1 + (y2-y1)*label_pos
                
                ax.text(label_x, label_y, 
                       f"w={network.weights[i][j,k]:.2f}",
                       fontsize=6, ha='center', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def main():
    # Create network and sample input
    network = NeuralNetwork([3, 4, 2])
    input_values = [0.5, 0.3, 0.8]  # Example inputs
    
    # Set up figure
    fig = plt.figure(figsize=(18, 9))
    ax = fig.gca()
    ax.axis('off')
    
    # Draw enhanced network
    draw_neural_net(ax, network, input_values)
    
    # Add title and info
    plt.title(f"Neural Network Visualization\nInput: {input_values}\n", pad=20)
    plt.text(0.5, -0.1, 
             f"Input Layer: {network.layers[0]} neurons | "
             f"Hidden Layer: {network.layers[1]} neurons | "
             f"Output Layer: {network.layers[2]} neurons",
             ha='center', va='center', transform=ax.transAxes)
    
    # Add legend
    plt.text(0.02, 0.98,
             "Blue: Biases\nBlack: Weights\nNode values: Activations",
             transform=ax.transAxes,
             ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('organized_network.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'organized_network.png'")
    plt.show()

if __name__ == "__main__":
    main()