import numpy as np

class Sgd():
    """
    Stocastic Gradient Descent (SGD), returns next updated tensor values by given learning rate
    y_n+1 = y_n - mu * d/dx.f(n)
     """
    def __init__(self, learning_rate):
        #Initialize constructor variables
        self.learning_rate = learning_rate 
        
    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_tensor = weight_tensor - self.learning_rate * gradient_tensor #Gradient Descent
        return updated_tensor
    

class SgdWithMomentum:
    """
    yet to add
    """
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = self.momentum_rate * self.velocity + gradient_tensor
        update = self.learning_rate * self.velocity
        return weight_tensor - update


class Adam:
    """
    Yet to add
    """
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.time = 0
        self.moment_1 = None
        self.moment_2 = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.time += 1

        if self.moment_1 is None:
            self.moment_1 = np.zeros_like(weight_tensor)
            self.moment_2 = np.zeros_like(weight_tensor)

        self.moment_1 = self.mu * self.moment_1 + (1 - self.mu) * gradient_tensor
        self.moment_2 = self.rho * self.moment_2 + (1 - self.rho) * gradient_tensor**2

        corrected_bias_for_m1 = self.moment_1 / (1 - self.mu**self.time)
        corrected_bias_for_m2 = self.moment_2 / (1 - self.rho**self.time)

        result = self.learning_rate * corrected_bias_for_m1 / (np.sqrt(corrected_bias_for_m2) + 1e-8)
        
        return weight_tensor - result
