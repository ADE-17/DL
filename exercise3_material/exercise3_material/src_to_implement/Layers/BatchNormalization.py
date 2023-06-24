import numpy as np
import copy

class BatchNormalization:
    """
    Batch Normalization layer 
    """

    def __init__(self, channels):
        """
        Constructor.

        Args:
            channels (int): Number of channels in the input tensor.
        """
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.moving_mean = None
        self.moving_var = None
        self.moving_avg_decay = 0.8 #alpha

        #Initialize
        self.weights =  np.ones(self.channels) # gamma
        self.bias =  np.zeros(self.channels) # beta

    def forward(self, input_tensor):
        """
        Performs the forward pass for the Batch Normalization layer during the training phase.

        Args:
            input_tensor (array/tensor): The input tensor to the Batch Normalization layer.

        Returns:
            array/tensor: The output tensor after applying Batch Normalization during the training phase.
        """
        epsilon = 1e-15 #Set epi
        need_conv = False
        
        if input_tensor.ndim == 4: #Handle 3-D tensor by reformating input tensor 
            need_conv = True
            input_tensor = self.reformat(input_tensor)
            
        self.input_tensor = input_tensor
            
        if self.testing_phase:
            self.mean = self.moving_mean
            self.var = self.moving_var
        else:
            self.mean = np.mean(input_tensor, axis= 0)
            self.var = np.var(input_tensor, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_var = copy.deepcopy(self.var)
            else:
                self.moving_mean = self.moving_mean * self.moving_avg_decay + self.mean * (1 - self.moving_avg_decay)
                self.moving_var = self.moving_var * self.moving_avg_decay + self.var * (1 - self.moving_avg_decay)
                
        self.input_tensor_hat = (input_tensor - self.mean) / np.sqrt(self.var + epsilon)
        output_tensor = self.weights * self.input_tensor_hat + self.bias
        
        if need_conv: #handle output of 3-D tensor by reformating output tensor
            output_tensor = self.reformat(output_tensor)
            
        return output_tensor

    def reformat(self, input_tensor):
        if input_tensor.ndim == 4:
            self.reformat_shape = input_tensor.shape
            B, H, M, N = input_tensor.shape
            input_tensor = input_tensor.reshape(B, H, M * N)
            input_tensor = input_tensor.transpose(0, 2, 1)
            input_tensor = input_tensor.reshape(B * M * N, H)
            return input_tensor
        else:
            B, H, M, N = self.reformat_shape
            input_tensor = input_tensor.reshape(B, M * N, H)
            input_tensor = input_tensor.transpose(0, 2, 1)
            input_tensor = input_tensor.reshape(B, H, M, N)
            return input_tensor