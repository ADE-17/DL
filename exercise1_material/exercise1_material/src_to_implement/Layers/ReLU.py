import numpy as np

class ReLU:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output

    def backward(self, error_tensor):
        output_tensor = np.copy(error_tensor)
        output_tensor[self.input_tensor <= 0] = 0
        return output_tensor