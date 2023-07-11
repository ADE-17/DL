import numpy as np
from Layers import Base

class Pooling(Base.BaseLayer):
    """
    Pooling layer: Max pooling layer implemented which reduces dimension of the hidden layer
    Inherits the base layer
    """
    def __init__(self, stride_shape, pooling_shape):
        """Constructer
        Args:
            stride_shape (single_value or tuple): Stride is the number of pixels shifts over the input matrix
            pooling_shape (tuple): Pooling kernel/window determines the size of the window used for pooling
        """
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        """Foward pass of p1 max pooling layer
        Performs pooling by moving p1 window across the input tensor and selecting the maximum value within each window. 
        It keeps track of the positions of these maximum values and stores them in auxiliary arrays. 
        Finally, it constructs the output tensor by assigning the maximum values to their corresponding positions.

        Args:
            input_tensor (tensor): input tensor 

        Returns:
            tensor: output tensor
        """
        self.save_shape = input_tensor.shape #saving shape, to be used in backward pass
        
        #input_tensor.shape[2]: number of rows in input tensor (height)
        #pooling_shape[0]: height of pooling window
        #stride_shape[0]: vertical stride (the number of pixels the pooling window moves vertically)
        #horz_pool_no = number of horizontal pooling operations
        horz_pool_no = np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0]) # determines how many vertical strides are needed to cover the entire height of the input tensor.
        
        #input_tensor.shape[3]: number of columns in the input tensor (width)
        #pooling_shape[1]: width of the pooling window
        #stride_shape[1]: horizontal stride (the number of pixels the pooling window moves horizontally)
        #vert_pool_no = number of verticle pooling operations
        vert_pool_no = np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1]) #determines how many horizontal strides are needed to cover the entire width of the input tensor
        
        #initialize with zeros
        
        #output_tensor: store pooled values
        #max_x_indices: stores the x_indices indices of the maximum values within each pooling window
        #max_y_indices: stores the y_indices indices of the maximum values within each pooling window
        output_tensor = np.zeros((*input_tensor.shape[0:2], int(horz_pool_no), int(vert_pool_no))) 
        self.max_x_indices = np.zeros((*input_tensor.shape[0:2], int(horz_pool_no), int(vert_pool_no)), dtype=int)
        self.max_y_indices = np.zeros((*input_tensor.shape[0:2], int(horz_pool_no), int(vert_pool_no)), dtype=int)
        
        #iterates over the horizontal and vertical positions of the input tensor to perform pooling
        #temp: flattened pooling window
        #output_indices: stores the indices of the maximum values within the pooling window
        p1 = 0  # index for the horizontal dimension
        for p3 in range(0, input_tensor.shape[2] - self.pooling_shape[0] + 1, self.stride_shape[0]):
            
            p2 = 0  # index for the verticle dimension
            
            for p4 in range(0, input_tensor.shape[3] - self.pooling_shape[1] + 1, self.stride_shape[1]):
                
                temp = input_tensor[:, :, p3:p3+self.pooling_shape[0], p4:p4+self.pooling_shape[1]].reshape(*input_tensor.shape[0:2], -1)
                output_indices = np.argmax(temp, axis=2)
                
                x_indices = output_indices // self.pooling_shape[1] #row indices
                y_indices = output_indices % self.pooling_shape[1] #column indices
                
                self.max_x_indices[:, :, p1, p2] = x_indices #indices are assigned
                self.max_y_indices[:, :, p1, p2] = y_indices
                
                output_tensor[:, :, p1, p2] = np.choose(output_indices, np.moveaxis(temp, 2, 0))
                
                
                p2 += 1  # Increment p2 by 1
                
            p1 += 1  # Increment p1 by 1 
                
        return output_tensor
    
    def backward(self, error_tensor):
        """
        Backward pass of a max pooling layer
        Distributes the gradients from the error_tensor to their original positions 
        in the input tensor based on the indices stored during the forward pass.allows the gradients to 
        flow back to the appropriate locations in the previous layer during backpropagation
        Args:
            backpass_output (_type_): _description_

        Returns:
            tensor: error tensor, accumulated gradients is returned
        """
        backpass_output = np.zeros(self.save_shape) # initialize with zeros using saved shape of input_tensor
        
        #iterates through each dimension of stored max_x and max_y indices
        # The error value at (a, b, i, j) is added to the computed position in the return_tensor. 
        # This accumulates the gradients at the correct locations.
        for p1 in range(self.max_x_indices.shape[0]):
            for p2 in range(self.max_x_indices.shape[1]):
                for p3 in range(self.max_x_indices.shape[2]):
                    for p4 in range(self.max_y_indices.shape[3]):
                        backpass_output[p1, p2, p3*self.stride_shape[0] + self.max_x_indices[p1, p2, p3, p4], 
                                        p4*self.stride_shape[1]+self.max_y_indices[p1, p2, p3, p4]] += error_tensor[p1, p2, p3, p4]
                        
        return backpass_output
    
    # def forward(self, input_tensor):
    #     h_pools = int((input_tensor.shape[2] - self.pooling_shape[0]) / self.stride_shape[0]) + 1
    #     v_pools = int((input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1

    #     output_tensor = np.zeros((*input_tensor.shape[0:2], h_pools, v_pools))
    #     self.max_x_indices = np.zeros((*input_tensor.shape[0:2], h_pools, v_pools), dtype=int)
    #     self.max_y_indices = np.zeros((*input_tensor.shape[0:2], h_pools, v_pools), dtype=int)

    #     # Create row and column indices for the pooling windows
    #     rows = np.arange(self.pooling_shape[0])
    #     cols = np.arange(self.pooling_shape[1])

    #     # Compute the indices for each pooling window
    #     row_indices = rows.reshape(-1, 1) + self.stride_shape[0] * np.arange(h_pools)
    #     col_indices = cols.reshape(-1, 1) + self.stride_shape[1] * np.arange(v_pools)

    #     # Extract the pooling windows from the input tensor
    #     pooling_windows = input_tensor[:, :, row_indices, col_indices]

    #     # Reshape the pooling windows for comparison
    #     reshaped_windows = pooling_windows.reshape(*input_tensor.shape[0:2], -1)

    #     # Find the indices of the maximum values within each pooling window
    #     output_indices = np.argmax(reshaped_windows, axis=2)

    #     # Calculate the x_indices and y_indices indices within the pooling window
    #     x_indices = output_indices // self.pooling_shape[1]
    #     y_indices = output_indices % self.pooling_shape[1]

    #     # Update the max_x_indices and max_y_indices arrays
    #     self.max_x_indices[:, :, :, :] = x_indices[:, :, np.newaxis, np.newaxis]
    #     self.max_y_indices[:, :, :, :] = y_indices[:, :, np.newaxis, np.newaxis]

    #     # Choose the maximum values for each pooling window and assign to the output tensor
    #     output_tensor[:, :, :, :] = np.choose(output_indices, np.moveaxis(reshaped_windows, 2, 0))

    #     return output_tensor
