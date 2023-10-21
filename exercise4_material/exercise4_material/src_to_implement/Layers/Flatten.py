class Flatten:
    """
    Implemented a flatten layer
    Convert all the resultant 2-Dimensional arrays from pooled feature maps into a single 
    long continuous linear vector
    """
    def __init__(self):
        """
        Constructer 
        """
        self.trainable = False

    def forward(self, input_tensor):
        """
        Perform the forward pass of the flatten layer
        Args:
            input_tensor (array/tensor): batch of multi-dimensional arrays (spatial + channels)
            
        Returns:
            array/tensor: batch of one dimensional feature vectors
        """
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, error_tensor):
        """
        Perform the backward pass of the flatten layer.
        
        Args:
            error_tensor (array/tensor): The gradient of the loss function with respect to the output of the flatten layer.
            
        Returns:
            array/tensor: The gradient of the loss function with respect to the input of the flatten layer, reshaped to match the original input shape.
        """
        return error_tensor.reshape(self.input_shape)
