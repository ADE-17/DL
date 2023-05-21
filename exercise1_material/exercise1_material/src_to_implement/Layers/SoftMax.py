import numpy as np

class SoftMax:
    def __init__(self):
        self.trainable = False
        self.output = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output /= np.sum(self.output, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        error_back = self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))
        return error_back
