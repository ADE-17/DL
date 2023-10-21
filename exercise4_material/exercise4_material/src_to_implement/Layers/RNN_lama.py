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

        # initialize d_hidden with zeros
        d_hidden = np.zeros((self.output_size,))
        self.gradient_inputs = np.zeros((time_steps, self.input_size))

        # reversing the sequence
        for t in reversed(range(time_steps)):
            print("dhidden.shape:  ", d_hidden.shape)
            print("error_tensor[t].shape  ", error_tensor[t].shape)

            d_output_t = error_tensor[t] + d_hidden

            # backpropagate through FCY and Sigmoid
            self.fcy.input = self.outputs[t].reshape(1,-1)  # Update input of fcy for each timestep
            dhidden = self.fcy.backward(d_output_t.reshape(1,-1))  # It updates self.fcy.weights and returns d_hidden
            self.gradient_why = self.fcy._gradient_weights
            Sigmoid.activations = self.outputs[t]
            self.gradient_output = Sigmoid.backward(self, d_hidden)
            TanH.activations = self.hiddenstates[t]
            # prepare for backpropagation through FCW and Tanh
            print("self.hiddenstates[t]", (1-np.square(self.hiddenstates[t])).shape)
            print("dhidden", dhidden.shape)
            print("grad out", self.gradient_output.shape)

            d_hidden = self.gradient_output.T * TanH.backward(self, dhidden)

            # concatenate the hidden state and input
            xs = np.concatenate((self.hiddenstates[t - 1], self.inputs[t]), axis=0) if t > 0 else self.inputs[t]

            # backpropagate through FCW
            self.fcw.input = xs  # Update input of fcw for each timestep
            d_hidden = self.fcw.backward(d_hidden)  # It updates self.fcw.weights and returns d_hidden
            self._gradient_weights += self.fcw._gradient_weights
            # split d_hidden into d_hidden and gradient_inputs
            if t > 0:
                self.gradient_inputs[t], d_hidden = np.split(d_hidden, [self.input_size], axis=0)
            else:
                self.gradient_inputs[t] = d_hidden[:self.input_size]

        return self.gradient_inputs

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
