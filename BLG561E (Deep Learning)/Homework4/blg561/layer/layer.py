
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
        pass

    def backward(self, dprev):
        pass


class Tanh(Layer):
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x.copy()
        x = x.copy()
        return np.tanh(x)

    def backward(self, dprev):
        return (1 - np.tanh(self.x) ** 2) * dprev


class Softmax(Layer):
    def __init__(self):
        self.probs = None

    def forward(self, x):
        '''
            Softmax function
            :param x: Input for classification (Likelihoods)
            :return: Class Probabilities
        '''
        pass

    def backward(self, y):
        '''
            Implement the backward pass w.r.t. softmax loss
            -----------------------------------------------
            :param y: class labels. (as an array, [1,0,1, ...]) Not as one-hot encoded
            :return: upstream derivate

        '''
        pass


def loss(probs, y):
    '''
        Calculate the softmax loss
        --------------------------
        :param probs: softmax probabilities
        :param y: correct labels
        :return: loss
    '''
    pass


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
        pass

    def backward(self, dprev):

       pass


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
        pass

    def backward(self, dprev):
        pass


class MaxPool2d(Layer):
    def __init__(self, pool_height, pool_width, stride):
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.stride = stride
        self.x = None

    def forward(self, x):
       pass

    def backward(self, dprev):
        pass


class Flatten(Layer):
    def __init__(self):
        self.N, self.C, self.H, self.W = 0, 0, 0, 0

    def forward(self, x):
        self.N, self.C, self.H, self.W = x.shape
        out = x.reshape(self.N, -1)
        return out

    def backward(self, dprev):
        return dprev.reshape(self.N, self.C, self.H, self.W)
