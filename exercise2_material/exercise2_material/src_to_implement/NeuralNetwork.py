import numpy as np
import copy


class NeuralNetwork:
    """
    NeuralNetwork representing architecture of Neural-Network.
    """

    def __init__(self, optimizer, weight_initializer=None, bias_initializer=None):
        # Initialize constructor variables
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        """
        Takes input from data layer and passes it through all layers in the neural network.
        """
        input_tensor, self.label_tensor = self.data_layer.next()
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        output = self.loss_layer.forward(output, self.label_tensor)
        return output

    def backward(self, label_tensor):
        """
        Inputs labels and propagates it back through the network.
        """
        error = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        """
        Stacks both trainable/non-trainable layers to the network.
        Initializes trainable layers with stored initializers.
        """
        if layer.trainable:
            layer.weights = self.weight_initializer(layer.weights.shape)
            layer.biases = self.bias_initializer(layer.biases.shape)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        """
        Train network and store loss for each iteration.
        """
        for iteration in range(iterations):
            output = self.forward()
            self.loss.append(output)
            self.backward(self.label_tensor)

    def test(self, input_tensor):
        """
        Propagates input through the network and returns prediction of the last layer.
        """
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output
