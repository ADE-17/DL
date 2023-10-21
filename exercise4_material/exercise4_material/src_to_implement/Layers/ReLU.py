import numpy as np

class ReLU:
    """
    Rectified Linear Unit (RELU) for Neural-Network, implements a forward and backward pass
    taking input_tensor and error_tensor to return ReLU output (positive part of its argument)
    """
    def __init__(self):
        self.trainable = False #Set false

    def forward(self, input_tensor):
        """
        ReLU Activation function: f(x) = max(0, input)
        """
        self.input_tensor = input_tensor
        self.output = np.maximum(0, input_tensor)
        return self.output

    def backward(self, error_tensor):
        """
        error_n-1 = 0; if input =< 0
                  = error_n; else
        """
        output_tensor = np.copy(error_tensor)
        output_tensor[self.input_tensor <= 0] = 0
        return output_tensor