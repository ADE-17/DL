import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, size=(input_size, output_size))
        self.optimizer = None

    def forward(self, input_tensor):
        # Perform forward pass calculations and return the output tensor
        output_tensor = np.dot(input_tensor, self.weights)
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def backward(self, error_tensor):
        # Perform backward pass calculations and return the error tensor for the previous layer
        if self.optimizer is not None:
            gradient_weights = np.dot(input_tensor.T, error_tensor)
            self.weights = self.optimizer.calculate_update(self.weights, gradient_weights)

        return np.dot(error_tensor, self.weights.T)

    @property
    def gradient_weights(self):
        return np.dot(input_tensor.T, error_tensor)
