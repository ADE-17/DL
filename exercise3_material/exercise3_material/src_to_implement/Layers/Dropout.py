import numpy as np

class Dropout:
    """
    Dropout layer
    """

    def __init__(self, probability):
        """
        Constructor

        Args:
            probability (float): The fraction of units to keep during training.
        """
        self.probability = probability
        self.temp_mask = None
        self.testing_phase = False
        self.trainable = False

    def forward(self, input_tensor):
        """
        Performs the forward pass for the Dropout layer during the training phase.

        Args:
            input_tensor (array/tensor): input tensor 
        Returns:
            array/tensor: output tensor after dropout 
        """
        if self.testing_phase:
            self.temp_mask = np.ones(input_tensor.shape)
        else:
            self.temp_mask = (np.random.rand(*input_tensor.shape) < self.probability).astype(float)
            self.temp_mask /= self.probability
            
        output_tensor = input_tensor * self.temp_mask
        
        return output_tensor

    def backward(self, error_tensor):
        """
        Performs the backward pass for the Dropout layer during the training phase.

        Args:
            error_tensor (array/tensor): error tensor 
        Returns:
            array/tensor: error tensor after  dropout 
        """
        return error_tensor * self.temp_mask

    # def forward_test(self, input_tensor):
        
    #     output_tensor = input_tensor * self.probability
    #     return output_tensor
