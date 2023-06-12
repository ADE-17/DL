import numpy as np

class CrossEntropyLoss:
    """
    Cross Entropy Loss layer for the Neural-Network, which implements a forward 
    and backward pass taking predictions and labels to return CE loss
    """
    def forward(self, prediction_tensor, label_tensor):
        """
        loss = Σ -In(predictions + epsilon) 
        """ 
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        epsilon = np.finfo(prediction_tensor.dtype).eps #Set epsilon as smallest representable number
        loss = -np.sum(label_tensor * np.log(self.prediction_tensor + epsilon)) 
        return loss

    def backward(self, label_tensor):
        """
        error_tensor = - (labels / predictions + epsilon)
        """
        self.label_tensor = label_tensor
        epsilon = np.finfo(self.prediction_tensor.dtype).eps #Set epsilon as smallest representable number
        error = -(label_tensor / (self.prediction_tensor + epsilon)) 
        return error