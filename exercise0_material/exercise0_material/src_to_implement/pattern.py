import numpy as np
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, resolution, tile_size):
        #Initilize
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = np.zeros((self.resolution,self.resolution)) #create canvas
        
    def draw(self):
        indices = np.indices((self.resolution//self.tile_size, self.resolution//self.tile_size)) #get index for each coordinate
        tile_sum = indices[0] + indices[1] 
        checkers_bool = tile_sum % 2 != 0 #mask alternate elements
        checkers_array = checkers_bool.astype(int)
        scale_factor = self.resolution//checkers_array.shape[0] #scale array to checkboard resolution
        self.output = np.kron(checkers_array, np.ones((scale_factor, scale_factor))).astype(int) #scale using Kronecker product of two arrays
        
        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap='gray') #show
        plt.show()
        
class Circle:
    def __init__(self, resolution, radius, coordinates):
        #Initilize
        self.resolution = resolution
        self.radius = radius
        self.coordinates = coordinates
        self.output = np.zeros((self.resolution, self.resolution), dtype=int) #get 0 canvas

    def draw(self):
        x, y = self.coordinates #get circle coordinates
        x_max, y_max = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution)) #Draw canvas
        shift = np.sqrt((x_max - x) ** 2 + (y_max - y) ** 2) #euclidian distance
        self.output[shift <= self.radius] = 1 #mask every other point on canvas not inside radius
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray') #show
        plt.show()
        
class Spectrum:
    def __init__(self, resolution):
        #Initalise
        self.resolution = resolution
        self.output = np.zeros((self.resolution, self.resolution, 3)) #3D array RGB
        
    def draw(self):
        
        red_channel = np.linspace(0, 1, self.resolution) #Red channel
        green_channel = np.linspace(0, 1, self.resolution)[:, np.newaxis] #green channel #reshaped to have a column vector
        
        self.output[:, :, 0] = red_channel
        self.output[:, :, 1] = green_channel
        self.output[:, :, 2] = 1 - red_channel #Blue channel
        return self.output.copy()
        
    def show(self):
        plt.imshow(self.output) #show
        plt.axis('off')
        plt.show()