# import numpy as np

# class Pooling:
#     def __init__(self, stride_shape, pooling_shape):
#         self.stride_shape = stride_shape
#         self.pooling_shape = pooling_shape
#         self.trainable = None
#         self.mask = None
    
#     def forward(self, input):
#         batch_size, num_channels, input_height, input_width = input.shape
#         stride_height, stride_width = self.stride_shape
#         pool_height, pool_width = self.pooling_shape
        
#         output_height = int((input_height - pool_height) / stride_height) + 1
#         output_width = int((input_width - pool_width) / stride_width) + 1
        
#         self.mask = np.zeros(input.shape)
        
#         output = np.zeros((batch_size, num_channels, output_height, output_width))
        
#         for h in range(output_height):
#             for w in range(output_width):
#                 h_start = h * stride_height
#                 h_end = h_start + pool_height
#                 w_start = w * stride_width
#                 w_end = w_start + pool_width
                
#                 # Perform max pooling within the pooling region
#                 pool_region = input[:, :, h_start:h_end, w_start:w_end]
#                 max_values = np.max(pool_region, axis=(2, 3), keepdims=True)
                
#                 # Store the max values and their indices for backward pass
#                 mask = (pool_region == max_values)
#                 self.mask[:, :, h_start:h_end, w_start:w_end] = mask
#                 output[:, :, h, w] = max_values[:, :, 0, 0]
        
#         return output
    
#     def backward(self, d_output):
#         batch_size, num_channels, output_height, output_width = d_output.shape
#         stride_height, stride_width = self.stride_shape
#         pool_height, pool_width = self.pooling_shape
        
#         d_input = np.zeros(self.mask.shape)
        
#         for h in range(output_height):
#             for w in range(output_width):
#                 h_start = h * stride_height
#                 h_end = h_start + pool_height
#                 w_start = w * stride_width
#                 w_end = w_start + pool_width
                
#                 # Get the stored mask for the pooling region
#                 mask = self.mask[:, :, h_start:h_end, w_start:w_end]
                
#                 # Compute the gradients for the pooling region
#                 d_output_region = d_output[:, :, h, w][:, :, np.newaxis, np.newaxis]
#                 d_input_region = d_input[:, :, h_start:h_end, w_start:w_end]
                
#                 d_input_region += mask * d_output_region
                
#         return d_input