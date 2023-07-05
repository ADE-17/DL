import numpy as np
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid
from Layers.FullyConnected import FullyConnected
class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.fcy= FullyConnected(self.hidden_size, self.output_size)
        self.why = self.fcy.weights # np.random.uniform(size=(self.output_size, self.hidden_size))
        self.gradient_why = self.fcy._gradient_weights

        self.fcw = FullyConnected ( self.hidden_size + self.input_size, self.hidden_size)
        self.weights = self.fcw.weights #np.random.uniform(size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self._gradient_weights= self.fcw._gradient_weights

        self.hidden_state = np.zeros((1,hidden_size))
        self._memorize = False
        self.trainable = True
        self._optimizer = None

        self.hiddenstates = []
        self.outputs = []
        self.inputs = []
        self.bptt = 0

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        if isinstance(value, bool):
            self._memorize = value
        else:
            raise ValueError('memorize must be a boolean value.')

    @property
    def gradient_weights(self):
        # return the weights used in the hidden state computation
        return self.weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        # set new value to the weights used in the hidden state computation
        self.weights = value

    def reset_state(self):
        # Reset stored inputs, outputs and hidden states to empty lists
        self.hiddenstates = []
        self.outputs = []
        self.inputs = []
        # Reset the inputs of fcw and fcy to None
        self.fcw.input = None
        self.fcy.input = None
        # Reset hidden_state to zeros
        self.hidden_state = np.zeros((1, self.hidden_size))

    def forward(self, input_tensor):
        if(not self._memorize):
            self.reset_state()
        time_steps = input_tensor.shape[0]  # Consider the “batch” dimension as the “time”

        output_tensor = np.zeros((time_steps, self.output_size))

        for t in range(time_steps):
            self.hiddenstates.append(self.hidden_state)
            self.inputs.append(input_tensor[t])

            x = input_tensor[t].reshape(1,-1)
            xs = np.concatenate((self.hidden_state,x), axis=1)
            self.fcw.input= xs
            self.hidden_state = TanH.forward(self,self.fcw.forward(xs))
            self.weights = self.fcw.weights

            output_tensor[t] = Sigmoid.forward(self,self.fcy.forward(self.hidden_state))
            self.why = self.fcy.weights

            self.outputs.append(output_tensor[t])
        return output_tensor

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]
        
        # input_grads = []
        input_grads = np.zeros((time_steps, self.input_size))

        d_hidden = np.zeros((1, self.hidden_size))
        self.gradient_weights_w = np.zeros((self.hidden_size+self.input_size+1, self.hidden_size))

        grad_tanh = 1-self.hidden_state[::] ** 2
        
        count = 0
        
        for t in reversed(range(time_steps)):
            
            yw_error = self.fcy.backward(error_tensor[t][np.newaxis, :])
            self.fcy.input = self.hiddenstates[t]
            
            d_output_t = yw_error + d_hidden
            
            # TanH.activations = self.hiddenstates[t]
            grad_hidden = grad_tanh[t] * d_output_t
            
            xw_error = self.fcw.backward(grad_hidden)
            
            hidden_error = xw_error[:, 0:self.hidden_size]
            
            x_error = xw_error[:, self.hidden_size:(self.hidden_size + self.input_size + 1)]
            self.input_grads[t] = x_error
            
            con = np.hstack((self.hiddenstates[t], self.input_tensor[t],1))
            
            self.fcw.input_tensor = con[np.newaxis, :]
            
            if count <= self.bptt:
                self.why = self.fcy.weights
                self.weights = self.fcw.weights
                self.gradient_why = self.fcy.gradient_weights
                self.gradient_weights_w = self.fcw.gradient_weights
            count += 1
            
        if self.optimizer is not None:
            self.why = self.optimizer.calculate_update(self.why, self.gradient_why)
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights_w)
            self.fcy.weights = self.why
            self.fcw.weights = self.weights
        return self.input_grads


    def calculate_regularization_loss(self):
        # if no regularizer is set, just return 0
        if self.optimizer is None or self.optimizer.regularizer is None:
            return 0
        # otherwise, calculate the regularization loss based on the weights
        return self.optimizer.regularizer.norm(self.weights)

    def initialize(self, weights_initializer, bias_initializer):
        self.fcw.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.hidden_size)
        self.fcy.weights = weights_initializer.initialize(self.why.shape, self.hidden_size, self.output_size)
        self.why = self.fcy.weights
        self.weights = self.fcw.weights


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.layer = self
