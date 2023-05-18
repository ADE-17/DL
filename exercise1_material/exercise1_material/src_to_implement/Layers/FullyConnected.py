import numpy as np
from Layers.Base import BaseLayer

import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.biases = np.random.uniform(0, 1, (1, output_size))
        self._gradient_weights = None
        self._gradient_biases = None
        self._optimizer = None

    def forward(self, input_tensor):
        self.input = input_tensor
        self.output = np.dot(np.concatenate([input_tensor, np.ones((input_tensor.shape[0], 1))], axis=1), self.weights) + self.biases
        return self.output

    def backward(self, error_tensor):
        self._gradient_weights = np.dot(np.concatenate([self.input, np.ones((self.input.shape[0], 1))], axis=1).T, error_tensor)
        self._gradient_biases = np.sum(error_tensor, axis=0, keepdims=True)
        self.gradient_input = np.dot(error_tensor, self.weights[:-1].T)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.biases = self._optimizer.calculate_update(self.biases, self._gradient_biases)

        return self.gradient_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_biases(self):
        return self._gradient_biases
