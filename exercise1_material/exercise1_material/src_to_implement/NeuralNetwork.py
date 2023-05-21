import numpy as np
import copy
                                                                                                                              
class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        
    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output, label_tensor
    
    def backward(self, label_tensor):
        error = self.loss_layer.backward(label_tensor)
        for layer in reversed(self.layers):
            error = layer.backward(error)
        
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)
        
    def train(self, iterations):
        for _ in range(iterations):
            output = self.forward()
            self.loss.append(self.loss_layer.forward(output))
            self.backward(self.data_layer.next()[1])
            
    def test(self, input_tensor):
        output = input_tensor
        for layer in self.layers:
            output = layer.forward(output)
        return output