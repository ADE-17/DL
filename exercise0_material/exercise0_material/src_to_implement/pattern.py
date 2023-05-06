import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((self.resolution,self.resolution))
        
    def draw(self):
        indices = np.indices((self.resolution//self.tile_size, self.resolution//self.tile_size))
        tile_sum = indices[0] + indices[1]
        checkers_bool = tile_sum % 2 != 0
        checkers_array = checkers_bool.astype(int)
        scale_factor = self.resolution//checkers_array.shape[0]
        self.output = np.kron(checkers_array, np.ones((scale_factor, scale_factor))).astype(int)
        
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()
        
class Circle:
    def __init__(self, resolution, radius, coordinates):
        self.resolution = resolution
        self.radius = radius
        self.coordinates = coordinates
        self.output = np.zeros((self.resolution, self.resolution), dtype=int)

    def draw(self):
        x_max, y_max = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        x, y = self.coordinates
        shift = np.sqrt((x_max - x) ** 2 + (y_max - y) ** 2)
        self.output[shift <= self.radius] = 1
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()