import numpy as np

class SoftMax:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        exp_input = np.exp(input_tensor)
        sum_exp_input = np.sum(exp_input, axis=1, keepdims=True)
        probabilities = exp_input / sum_exp_input
        return probabilities

    def backward(self, error_tensor):
        return error_tensor
