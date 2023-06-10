# import numpy as np
# from scipy.ndimage import convolve

# class Conv:
#     def __init__(self, stride_shape, convolution_shape, num_kernels):
#         if isinstance(stride_shape, int):
#             stride_shape = (stride_shape, stride_shape)
#         self.stride_shape = stride_shape

#         if len(convolution_shape) == 2:
#             self.convolution_shape = (1,) + convolution_shape
#         elif len(convolution_shape) == 3:
#             self.convolution_shape = convolution_shape
#         else:
#             raise ValueError("Invalid convolution shape. It should be either (c, m) for 1D or (c, m, n) for 2D.")

#         self.num_kernels = num_kernels
#         self.trainable = True
#         self.weights = np.random.rand(num_kernels, *self.convolution_shape)
#         self.bias = np.random.rand(num_kernels, 1)
#         self.weights_gradient = None
#         self.bias_gradient = None
#         self.optimizer = None
#         self.optimizer_bias = None

#     @property
#     def gradient_weights(self):
#         return self.weights_gradient

#     @property
#     def gradient_bias(self):
#         return self.bias_gradient

#     def forward(self, input_tensor):
#         batch_size, input_channels, input_height, input_width = input_tensor.shape
#         _, _, kernel_height, kernel_width = self.weights.shape
#         stride_height, stride_width = self.stride_shape

#         output_height = (input_height - kernel_height) // stride_height + 1
#         output_width = (input_width - kernel_width) // stride_width + 1

#         self.output_tensor = np.zeros((batch_size, self.num_kernels, output_height, output_width))

#         if kernel_width == 1 and stride_width == 1:
#             # Handle 1x1 convolutions efficiently
#             self.output_tensor = np.sum(input_tensor[:, :, :, None] * self.weights[:, :, None], axis=1)
#         elif len(self.convolution_shape) == 2:
#             # Handle 1D convolutions using correlation
#             for b in range(batch_size):
#                 for c_out in range(self.num_kernels):
#                     for i in range(output_height):
#                         receptive_field = input_tensor[b, :, i * stride_height:i * stride_height + kernel_height]
#                         self.output_tensor[b, c_out, i] = np.sum(receptive_field * self.weights[c_out]) + self.bias[c_out]
#         else:
#             # Perform 2D convolutions using scipy's n-dimensional convolution
#             for b in range(batch_size):
#                 for c_out in range(self.num_kernels):
#                     for c_in in range(input_channels):
#                         input_slice = input_tensor[b, c_in]
#                         kernel = self.weights[c_out, c_in]
#                         self.output_tensor[b, c_out] += convolve(input_slice, np.rot90(kernel, 2),
#                                                                  mode='constant', cval=0.0)[
#                             ::stride_height, ::stride_width]
#                     self.output_tensor[b, c_out] += self.bias[c_out]

#         return self.output_tensor

#     def backward(self, error_tensor):
#         batch_size, _, output_height, output_width = error_tensor.shape
#         _, _, kernel_height, kernel_width = self.weights.shape
#         stride_height, stride_width = self.stride_shape

#         self.weights_gradient = np.zeros_like(self.weights)
#         self.bias_gradient = np.sum(error_tensor, axis=(0, 2, 3), keepdims=True)

#         if len(self.convolution_shape) == 2:
#             # Handle 1D convolutions using correlation
#             input_height = (output_height - 1) * stride_height + kernel_height
#             input_tensor_gradient = np.zeros((batch_size, self.num_kernels, input_height))

#             for b in range(batch_size):
#                 for c_out in range(self.num_kernels):
#                     for i in range(output_height):
#                         receptive_field = input_tensor[b, :, i * stride_height:i * stride_height + kernel_height]
#                         self.weights_gradient[c_out] += receptive_field * error_tensor[b, c_out, i]
#                         input_tensor_gradient[b, :, i * stride_height:i * stride_height + kernel_height] += \
#                             self.weights[c_out] * error_tensor[b, c_out, i]

#             return input_tensor_gradient
#         else:
#             # Perform 2D convolutions using scipy's n-dimensional convolution
#             input_height = (output_height - 1) * stride_height + kernel_height
#             input_width = (output_width - 1) * stride_width + kernel_width
#             input_tensor_gradient = np.zeros((batch_size, input_tensor.shape[1], input_height, input_width))

#             for b in range(batch_size):
#                 for c_out in range(self.num_kernels):
#                     for c_in in range(input_tensor.shape[1]):
#                         input_slice = input_tensor[b, c_in]
#                         kernel_gradient = convolve(input_slice, error_tensor[b, c_out],
#                                                    mode='constant', cval=0.0, origin=1)
#                         self.weights_gradient[c_out, c_in] += kernel_gradient
#                         input_tensor_gradient[b, c_in] += convolve(error_tensor[b, c_out],
#                                                                    np.rot90(self.weights[c_out, c_in], 2),
#                                                                    mode='constant', cval=0.0)

#             return input_tensor_gradient

#     def initialize(self, weights_initializer, bias_initializer):
#         self.weights = weights_initializer.initialize(self.weights.shape)
#         self.bias = bias_initializer.initialize(self.bias.shape)

#         if self.optimizer is not None:
#             self.optimizer.initialize(self.weights.shape)
#         if self.optimizer_bias is not None:
#             self.optimizer_bias.initialize(self.bias.shape)
