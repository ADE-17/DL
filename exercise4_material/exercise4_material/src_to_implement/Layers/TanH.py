import numpy as np
from Layers import Base

class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__() # use instead of initializing trainable; for previous codes as well
        """
        TanH activation function.
        """
        self.activations = None

    def forward(self, input_tensor):
        """
        forward propagation using the TanH activation function.

        Args:
            input_tensor (tensor): input tensor.

        Returns:
            tensor: tensor with TanH activations.
        """
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        """
        backward propagation for TanH activation function.

        Args:
            error_tensor (tensor): error tensor from previous layer.

        Returns:
            tensor: gradient tensor for TanH activation function.
        """
        return error_tensor * (1 - np.square(self.activations))