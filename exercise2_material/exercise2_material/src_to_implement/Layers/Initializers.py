import numpy as np

class Constant:
    """
    Constant - A simple Initialization Scheme
    Initialized with default 0.1
    Returns: A new numpy array of given shape and filled with value given
    """
    def __init__(self, value=0.1):
        self.value = value
    
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.value)

class UniformRandom:
    """"
    UniformRandom -  A simple Initialization Scheme
    In range [0, 1]
    Returns: A random drawn from uniform distribution
    """
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(size=weights_shape)
    
class Xavier:
    """
    Xavier/Glorot Initializer
    σ = sqrt*(2/fan_in + fan_out)
    Keeps the variance of the activations and gradients roughly the same across layers.
    Returns: weights initialized using a normal distribution 
             with mean 0 and the calculated standard deviation.
    """ 
    def initialize(self, weights_shape, fan_in, fan_out):
        stddev = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(loc=0, scale=stddev, size=weights_shape)
    
class He:
    """
    He Initilizer
    σ = sqrt*(2/fan_in)
    Prevent the gradients from vanishing or exploding when using ReLU-based activations.
    Returns: weights initialized using a normal distribution 
             with mean 0 and the calculated standard deviation.
    """
    def initialize(self, weights_shape, fan_in, fan_out):
        stddev = np.sqrt(2 / fan_in)
        return np.random.normal(loc=0, scale=stddev, size=weights_shape)
