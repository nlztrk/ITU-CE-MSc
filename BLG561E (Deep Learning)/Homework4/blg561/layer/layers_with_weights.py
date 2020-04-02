#Adapted from CS231n
from .layer import Layer
from copy import deepcopy
from abc import abstractmethod
import numpy as np


class LayerWithWeights(Layer):
    '''
        Abstract class for layer with weights(CNN, Affine etc...)
    '''

    def __init__(self, input_size, output_size, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W = np.random.rand(input_size, output_size)
        self.b = np.zeros(output_size)
        self.x = None
        self.db = np.zeros_like(self.b)
        self.dW = np.zeros_like(self.W)

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed)

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
        pass

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        pass

    def __repr__(self):
        return 'Affine layer'


class Conv2d(LayerWithWeights):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        self.in_size = in_size
        self.out_size = out_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.x = None
        self.W = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.b = np.random.rand(out_size)
        self.db = np.random.rand(out_size, in_size, kernel_size, kernel_size)
        self.dW = np.random.rand(out_size)

    def forward(self, x):
        pass

    def backward(self, dprev):
        pass


class RNNLayer(LayerWithWeights):
    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        # self.W_ih = np.random.rand(in_size, out_size)
        # self.W_hh = np.random.rand(out_size, out_size)
        self.b = np.zeros(out_size)
        self.prev_H = []
        self.x = []
        self.db = None
        self.dW_ih = None
        self.dW_hh = None
        self.dprev_H = []

    def forward_step(self, x, prev_h=None):
        '''
            params
            x: input from the sequence
            prev_h: previous hidden state (it can be stored on class but it causes issue with numerical gradient checking)
        '''
        # Calculuate the input to hidden state
        i_h = x.dot(self.W_ih)
        # Calculate the hidden state
        h_h = prev_h.dot(self.W_hh)
        # Apply tanh activation
        next_h = np.tanh(i_h + h_h + self.b)
        # Store it for backward pass
        self.prev_H.append(next_h.copy())
        return next_h

    def forward(self, x, h0):
        '''
            params:
            x: sequence of input 
            h0: initial state
        '''

        N, T, D = x.shape
        out = np.zeros((N, T, self.out_size))
        prev_h = h0
        self.prev_H = [h0]
        for step in range(T):
            # Grab the corresponding sequence
            x_t = x[:, step, :]
            # Save it for gradient calculation
            self.x.append(x_t)
            # Iterate for once
            h = self.forward_step(x_t, prev_h)
            prev_h = h
            out[:, step, :] = h.copy()
        return out

    def backward_step(self, prev_h, dprev):
        '''
            params:
            prev_h: the previous hidden state before forward pass
            dprev: upstream gradient
        '''
        # HINT: You can access most recent activation by using self.prev_H[-1]
        # Think about how to access inputs utilizing the fact above.

        # In backward calculation, take the derivative of the RNN activation first (tanh for our case)
		
        most_rec_ac = self.prev_H[-1]
        lastx = self.x[-1]
		
        dtanh = (1 - most_rec_ac ** 2) * dprev
        
        self.dW_ih = np.dot(lastx.T, dtanh)
        self.dprev_H = np.dot(dtanh, self.W_hh.T)
        self.dW_hh = np.dot(prev_h.T, dtanh)  
        self.db = np.sum(dtanh, axis=0)
        dx = np.dot(dtanh, self.W_ih.T)
        
        return dx

    def backward(self, dprev):
        '''
            params:
            dprev: upstream gradient
            returns
            dx: downstream gradient
            gradient of first hidden state
            dw_ih: gradient of i-h connection
            dw_hh: gradient of h-h connection
            db: gradient of bias term
        '''

        N, T, H = dprev.shape
        D, _ = self.W_ih.shape
        dx = np.zeros((N, T, D))
        dW_hh = np.zeros_like(self.W_hh)
        dW_ih = np.zeros_like(self.W_ih)
        db = np.zeros_like(self.b)
        dprev_H = np.zeros_like(self.prev_H[0])

        for step in reversed(range(T)):
            curr_dprev = dprev_H + dprev[:, step, :]
            dx[:, step, :] = self.backward_step(self.prev_H[step], curr_dprev)

            dprev_H = self.dprev_H

            # Accumulate gradients here
            dW_hh += self.dW_hh
            dW_ih += self.dW_ih
            db += self.db

            # Remove them from the
            self.x.pop()
            self.prev_H.pop()

        return dx, self.dprev_H, dW_ih, dW_hh, db
