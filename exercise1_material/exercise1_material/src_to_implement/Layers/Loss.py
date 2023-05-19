import numpy as np

class CrossEntropyLoss:
    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        epsilon = np.finfo(prediction_tensor.dtype).eps
        
        loss = -np.sum(label_tensor * np.log(self.prediction_tensor + epsilon)) 
        return loss

    def backward(self, label_tensor):
        self.label_tensor = label_tensor
        epsilon = np.finfo(self.prediction_tensor.dtype).eps
        error = -(label_tensor / (self.prediction_tensor + epsilon)) 
        return error