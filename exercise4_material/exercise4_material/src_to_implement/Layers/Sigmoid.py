import numpy as np
from Layers import Base

class Sigmoid(Base.BaseLayer):
    """
    Sigmoid activation function
    """
    def __init__(self):
        super().__init__() # use instead of initializing trainable; for previous codes as well
        """Constructer
        Specifically store activations for the dynamic programming component
        """
        self.activations = None

    def forward(self, input_tensor):
        """foward propagation using sigmoid activation function

        Args:
            input_tensor (tensor): input tensor

        Returns:
            tensor: tensor with sigmoid activations
        """
        self.activations = 1 / (1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        """Backward pass

        Args:
            error_tensor (tensor): error tensor from previous layer

        Returns:
            tensor: gradiant tensor for sigmoid activation function
        """
        return error_tensor * self.activations * (1 - self.activations)
