import numpy as np

class NeuralNetwork:
    """
    A simple artificial neural network implementation for forward propagation.
    
    Attributes:
        layers (list): List of layer sizes (input, hidden layers, output)
        weights (list): List of weight matrices between layers
        biases (list): List of bias vectors for each layer
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with given layer sizes.
        
        Args:
            layer_sizes (list): List of integers representing the number of 
                                neurons in each layer (including input and output)
        """
        self.layers = layer_sizes
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases with random values
        for i in range(len(layer_sizes) - 1):
            # Weights between layer i and i+1
            weight_matrix = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            self.weights.append(weight_matrix)
            
            # Biases for layer i+1
            bias_vector = np.random.randn(layer_sizes[i+1])
            self.biases.append(bias_vector)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def forward_propagate(self, inputs):
        """
        Perform forward propagation through the network.
        
        Args:
            inputs (array-like): Input values to the network
            
        Returns:
            tuple: (final_output, layer_outputs) where final_output is the network's prediction
                   and layer_outputs contains outputs of all hidden layers
        """
        if len(inputs) != self.layers[0]:
            raise ValueError(f"Input size must match network input layer size ({self.layers[0]})")
            
        current_values = np.array(inputs)
        layer_outputs = []
        
        # Propagate through each layer
        for i in range(len(self.weights)):
            # Linear transformation: W*x + b
            current_values = np.dot(self.weights[i], current_values) + self.biases[i]
            
            # Apply activation function (sigmoid in this case)
            current_values = self.sigmoid(current_values)
            
            # Store hidden layer outputs (not including input or final output)
            if i < len(self.weights) - 1:
                layer_outputs.append(current_values)
        
        return current_values, layer_outputs


