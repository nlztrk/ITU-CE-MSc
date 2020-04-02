#Adapted from Stanford CS231n Course
import numpy as np
from abc import ABC, abstractmethod
from .helpers import flatten_unflatten


class Layer(ABC):
    def __init__(self, input_size, output_size):
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


class ReLU(Layer):

    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            Forward pass for ReLU
            :param x: outputs of previous layer
            :return: ReLU activation
        '''
            
        # Do not forget to copy the output to object to use it in backward pass
        self.x = x.copy()
        
        # This is used for avoiding the issues related to mutability of python arrays
        
        x = x.copy()
        
        out = np.maximum(0, x) # element-wise max operation
	
        # Implement relu activation
        return out

    def backward(self, dprev):
        '''
            Backward pass of ReLU
            :param dprev: gradient of previos layer:
            :return: upstream gradient
        '''
        # Your implementation starts
        
        if len(dprev)==3:
            dprev = dprev[0]
            
        dx = dprev * (self.x>0).astype(int) ## mask the values greater than zero, than transform True to 1, False to 0
         
        # End of your implementation
        
        return dx


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        # Normalize the class scores (i.e output of affine linear layers)
        # In order to avoid numerical unstability.
        # Do not forget to copy the output to object to use it in backward pass       
       
        # Your implementation starts
        
        expterm = np.exp(x - np.max(x, axis=1, keepdims=True)) # nominator
        probs = expterm / np.sum(expterm, axis=1, keepdims=True) # denominator
        
        probs += 1e-15 ## to prevent division by zero in zero probability situation
        
        
        self.probs = probs.copy()

        # End of your implementation
        return probs

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        # Your implementation starts

        diff_one_hot = np.zeros(self.probs.shape)
        diff_one_hot[np.arange(y.shape[0]), y] = -1
        dx = (diff_one_hot + self.probs) / y.shape[0]
        
        # End of your implementation

        return dx


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    # Your implementation starts
    
    elem_count = y.shape[0] # get element count

    true_class_probs = probs[np.arange(elem_count), y] ## only look at each element's true class probability
    
    loss = -np.sum(np.log(true_class_probs)) / elem_count ## compute the mean of the value, multiply with the minus since there is a log operation
    
    # End of your implementation
    
    return loss


class Dropout(Layer):
    def __init__(self, p=.5):
        '''
            :param p: dropout factor
        '''
        self.mask = None
        self.mode = 'train'
        self.p = p

    def forward(self, x, seed=None):
        '''
            :param x: input to dropout layer
            :param seed: seed (used for testing purposes)
        '''
        if seed is not None:
            np.random.seed(seed)

        if self.mode == 'train':

            # Create a dropout mask
            mask = (np.random.rand(*x.shape) > self.p) / self.p ## unpack the input shape and create random elimination mask

            # Do not forget to save the created mask for dropout in order to use it in backward
            self.mask = mask.copy()

            out = x * mask ## apply the mask to the input

            return out
            
        elif self.mode == 'test':
        
            out = x ## don't apply the mask to the input
            
            return out

        else:
            raise ValueError('Invalid argument!')

    def backward(self, dprev):
    
        if self.mode == 'train':
   
            dx = dprev * self.mask ## apply the mask during backprop
            
        elif self.mode == 'test':  
        
            dx = dprev ## don't apply the mask
              
        return dx


class BatchNorm(Layer):
    def __init__(self, D, momentum=.9):
        self.mode = 'train'
        self.normalized = None

        self.x_sub_mean = None
        self.momentum = momentum
        self.D = D
        self.running_mean = np.zeros(D)
        self.running_var = np.zeros(D)
        self.gamma = np.ones(D)
        self.beta = np.zeros(D)
        self.ivar = np.zeros(D)
        self.sqrtvar = np.zeros(D)

    # @flatten_unflatten
    def forward(self, x, gamma=None, beta=None):
        if self.mode == 'train':
        
            if len(x.shape) == 2:
                sample_mean = np.mean(x, axis=0)
                sample_var = np.var(x, axis=0)
                if gamma is not None:
                    self.gamma = gamma.copy()
                if beta is not None:

                    self.beta = beta.copy()

                # Normalise our batch
                self.normalized = ((x - sample_mean) /
                                   np.sqrt(sample_var + 1e-5)).copy()
                self.x_sub_mean = x - sample_mean


                # Update our running mean and variance then store.

                running_mean = np.zeros(x.shape[1])
                running_var = np.zeros(x.shape[1])
              
                
                running_mean = self.momentum * running_mean + (1 - self.momentum) * sample_mean 
                running_var = self.momentum * running_var + (1 - self.momentum) * sample_var
                
                out =  self.gamma * self.normalized + self.beta
                
                ## COMMENTIT
                
                self.running_mean = running_mean.copy()
                self.running_var = running_var.copy()

                self.ivar = 1./np.sqrt(sample_var + 1e-5)
                self.sqrtvar = np.sqrt(sample_var + 1e-5)

                return out
                
            elif len(x.shape) == 4:
                # extract the dimensions
                N, C, H, W = x.shape
                # mini-batch mean
                sample_mean = np.mean(x, axis=(0, 2, 3))
                # mini-batch variance
                sample_var = np.mean((x - sample_mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
                
                if gamma is not None:
                    self.gamma = gamma.copy()
                if beta is not None:
                    self.beta = beta.copy()
                    
                # normalize
                self.normalized = (x - sample_mean.reshape((1, C, 1, 1))) * 1.0 / np.sqrt(sample_var.reshape((1, C, 1, 1)) + 1e-5)
                self.x_sub_mean = x - sample_mean.reshape((1, C, 1, 1))
                # scale and shift
                
                running_mean = np.zeros(x.shape[1])
                running_var = np.zeros(x.shape[1])                
                running_mean = self.momentum * running_mean + (1 - self.momentum) * sample_mean 
                running_var = self.momentum * running_var + (1 - self.momentum) * sample_var
                    
                out = self.gamma.reshape((1, C, 1, 1)) * self.normalized + self.beta.reshape((1, C, 1, 1))      
                self.running_mean = running_mean.copy()
                self.running_var = running_var.copy()

                self.ivar = 1./np.sqrt(sample_var + 1e-5)
                self.sqrtvar = np.sqrt(sample_var + 1e-5)   
                
                return out
                   
        elif self.mode == 'test':
        
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])
            self.normalized = ((x - self.running_mean) /
                               np.sqrt(self.running_var + 1e-5)).copy()
                               
            out =  self.gamma * self.normalized + self.beta
            
        else:
            raise Exception(
                "INVALID MODE! Mode should be either test or train")
        return out

    def backward(self, dprev):
    
        if len(dprev.shape) == 2:
            N, D = dprev.shape

            dbeta = np.sum(dprev, axis=0)
            dgamma = np.sum(dprev*self.normalized, axis=0)
            dx = dprev * self.gamma
            dx_mu_01 = dx*self.ivar
            divar = np.sum(dx*self.x_sub_mean,axis = 0)
            dsqrtvar = -divar / self.sqrtvar**2
            dvar = (0.5/self.sqrtvar)*dsqrtvar
            dsq  = (1/N)*np.ones(dprev.shape)*dvar
            dx_mu_02 = 2*self.x_sub_mean*dsq
            d_mu = -1*np.sum(dx_mu_01+dx_mu_02, axis=0)
            d_x_01 = dx_mu_01 + dx_mu_02
            d_x_02 = (1/N)*np.ones(dprev.shape)*d_mu
            dx = d_x_01 + d_x_02
            # Calculate the gradients

            return dx, dgamma, dbeta
            
        if len(dprev.shape) == 4:
            N, C, H, W = dprev.shape

            dbeta = np.sum(dprev, axis=0)
            dgamma = np.sum(dprev*self.normalized, axis=0)
            dx = dprev * self.gamma.reshape((1, C, 1, 1))
            dx_mu_01 = dx*self.ivar.reshape((1, C, 1, 1))
            divar = np.sum(dx*self.x_sub_mean,axis = 0)
            dsqrtvar = -divar / (self.sqrtvar**2).reshape((C, 1, 1))
            dvar = (0.5/self.sqrtvar.reshape((C, 1, 1)))*dsqrtvar
            dsq  = (1/N)*np.ones(dprev.shape)*dvar
            dx_mu_02 = 2*self.x_sub_mean*dsq
            d_mu = -1*np.sum(dx_mu_01+dx_mu_02, axis=0)
            d_x_01 = dx_mu_01 + dx_mu_02
            d_x_02 = (1/N)*np.ones(dprev.shape)*d_mu
            dx = d_x_01 + d_x_02
            # Calculate the gradients

            return dx, dgamma, dbeta

class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_H = np.int(((H - self.pool_height) / self.stride) + 1)
        out_W = np.int(((W - self.pool_width) / self.stride) + 1)

        self.x = x.copy()

        # Initiliaze the output
        out = np.zeros([N, C, out_H, out_W])

        for idx in range(N): ## iterating over samples
        
            for h in range(0, out_H): ## iterating over height
            
                for w in range(0, out_W): ## iterating over width
                
                    out[idx, :, h, w] = np.amax(x[idx, :, h*self.stride:h*self.stride+self.pool_height, w*self.stride:w*self.stride+self.pool_width], axis=(-1, -2)) ## selects a region with a size pool width-height, selects maximum value in that region and outputs it

        return out

    def backward(self, dprev):
        x = self.x
        N, C, H, W = x.shape
        _, _, dprev_H, dprev_W = dprev.shape

        dx = np.zeros_like(self.x)

        for idx in range(N): ## iterating over samples
        
            for fidx in range(C): ## iterating over channels
            
                for h in range(dprev_H): ## iterating over height
                
                    for w in range(dprev_W): ## iterating over width
                    
                        ind = np.argmax(x[idx, fidx, h*self.stride:h*self.stride+self.pool_height, w*self.stride:w*self.stride+self.pool_width])
                        ind1, ind2 = np.unravel_index(ind, (self.pool_height, self.pool_width))
                        dx[idx, fidx, h*self.stride:h*self.stride+self.pool_height, w*self.stride:w*self.stride+self.pool_width][ind1, ind2] = dprev[idx, fidx, h, w]
                        ## COMMENTIT
                        
        return dx


class Flatten(Layer):
    def __init__(self):
        self.N, self.C, self.H, self.W = 0, 0, 0, 0

    def forward(self, x):
        self.N, self.C, self.H, self.W = x.shape
        out = x.reshape(self.N, -1)
        return out

    def backward(self, dprev):
        return dprev.reshape(self.N, self.C, self.H, self.W)
        
