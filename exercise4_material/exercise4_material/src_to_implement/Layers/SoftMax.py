 import numpy as np

class SoftMax:
    """
    SoftMax activation for Neural-Network to scale logits/input into probabilites, implements a forward
    and backward pass taking input_tensor and error_tensor to return probabilities of each outcome
    """
    def __init__(self):
        self.trainable = False #Set false
        self.output = None

    def forward(self, input_tensor):
        """
        Activation predcition (y_hat) for every element of batch: 
        y_k = exp(input) / Σ exp(input)
        """
        self.input_tensor = input_tensor
        self.output = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output /= np.sum(self.output, axis=1, keepdims=True) 
        return self.output

    def backward(self, error_tensor):
        """
        error_n-1 = prediction * (error_n - Σ error_n * prediction)
        """
        self.error_tensor = error_tensor
        error_back = self.output * (error_tensor - np.sum(error_tensor * self.output, axis=1, keepdims=True))
        return error_back
