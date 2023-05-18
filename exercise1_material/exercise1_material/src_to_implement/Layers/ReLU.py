import numpy as np

class ReLU:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input = input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output

    def backward(self, error_tensor):
        self.gradient_input = np.multiply(error_tensor, np.heaviside(self.input, 0))
        return self.gradient_input
