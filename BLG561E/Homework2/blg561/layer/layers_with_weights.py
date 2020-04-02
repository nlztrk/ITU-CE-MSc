#Adapted from Stanford CS231n Course

from .layer import Layer
from copy import copy
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
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)
        self.desc = "init"

    def forward(self, x):
        '''
            :param x: activations/inputs from previous layer
            :return: output of affine layer
        '''
          
        # Vectorize the input to [batchsize, others] array
        
        batch_size = x.shape[0]
           	
        x_vec = np.reshape(x, (batch_size, -1)) ## flatten except elem size value

        # Do the affine transform

        out = x_vec.dot(self.W) + self.b # wx+b
        	
        # Save x for using in backward pass
        
        self.x = x.copy()
	
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        
        if len(dprev)==3:
            dprev = dprev[0]
            
        batch_size = self.x.shape[0]
        
        # Vectorize the input to a 1D ndarray
        
        x_vectorized = np.reshape(self.x, (batch_size, -1)) ## flatten except elem size value

        # YOUR CODE STARTS

        db = np.sum(dprev, axis=0)
        
        dx_normal = dprev.dot(self.W.T)
        
        dx = dx_normal.reshape(self.x.shape)
        
        dw = x_vectorized.T.dot(dprev)
        
        # YOUR CODE ENDS

        # Save them for backward pass
        
        self.db = db.copy()
        self.dW = dw.copy()
        
        return dx, dw, db

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
        N, C, H, W = x.shape
        F, C, FH, FW = self.W.shape
        self.x = copy(x)
        # pad X according to the padding setting
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')

        # Calculate output's H and W according to your lecture notes
        out_H = np.int(((H + 2*self.padding - FH) / self.stride) + 1)
        out_W = np.int(((W + 2*self.padding - FW) / self.stride) + 1)

        # Initiliaze the output
        out = np.zeros([N, F, out_H, out_W])

        # TO DO: Do cross-correlation by using for loops
	
        for idx in range(N): ## sample iterator

            for fidx in range(F): ## filter iterator

                for h in range(out_H): ## height iterator

                    for w in range(out_W): ## width iterator

	                    weight_prod = padded_x[idx, :, h*self.stride:h*self.stride+FH, w*self.stride:w*self.stride+FW] * self.W[fidx, :, :, :] ## element-wise multiplication for convolution
	                    conv_sum = weight_prod.sum() ## sum all multiplications for convolution
	                    out[idx, fidx, h, w] = conv_sum + self.b[fidx] ## assign the value with bias addition as corresponding output
					
        return out

    def backward(self, dprev):
    
        if len(dprev)==3:
            dprev = dprev[0]
        dx, dw, db = None, None, None
        padded_x = np.pad(self.x, ((0, 0), (0, 0), (self.padding, self.padding),
                                   (self.padding, self.padding)), 'constant')
        N, C, H, W = self.x.shape
        F, C, FH, FW = self.W.shape
        _, _, out_H, out_W = dprev.shape

        dx_temp = np.zeros_like(padded_x).astype(np.float32)
        dw = np.zeros_like(self.W).astype(np.float32)
        db = np.zeros_like(self.b).astype(np.float32)
		
        dx = np.zeros_like(self.x).astype(np.float32)

        
        for idx in range(N): ## sample iterator
        
            for fidx in range(F): ## filter iterator
            
                db[fidx] += dprev[idx, fidx].sum() ## calculated bias gradient of selected filter
                
                for h in range(0, out_H): ## height iterator
                
                    for w in range(0, out_W): ## width iterator
                    
                        dw[fidx] += padded_x[idx, :, h * self.stride:h * self.stride + FH, w * self.stride:w * self.stride + FW] * dprev[idx, fidx, h, w] ## calculating weight gradient for the given filter
                        
                        dx_temp[idx, :, h * self.stride:h * self.stride + FH, w * self.stride:w * self.stride + FW] += self.W[fidx] * dprev[idx, fidx, h, w] ## creating padded version of the value gradient matrix (padded)
                        
                        
        dx = dx_temp[:, :, self.padding:self.padding+H, self.padding:self.padding+W] ## getting value gradient from the padded matrix

        self.db = db.copy()
        self.dW = dw.copy()
        return dx, dw, db
