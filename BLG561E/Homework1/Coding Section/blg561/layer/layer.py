import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    '''
        Abstract layer class which implements forward and backward methods
    '''

    def __init__(self):
        self.x = None

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError('Abstract class!')

    @abstractmethod
    def backward(self, x):
        raise NotImplementedError('Abstract class!')

    def __repr__(self):
        return 'Abstract layer class'


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


class ReLU(Layer):

    def __init__(self):
        # Dont forget to save x or relumask for using in backward pass
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
        
        dx = dprev * (self.x>0).astype(int) ## mask the values greater than zero, than transform True to 1, False to 0
         
        # End of your implementation
        
        return dx
        

class YourActivation(Layer): ## SWISH ACTIVATION FUNCTION
    def __init__(self):
        self.x = None

    def forward(self, x):
        '''
            :param x: outputs of previous layer
            :return: output of activation
        '''
        # Lets have an activation of X^2
        # TODO: CHANGE IT
        self.x = x.copy()
        out = self.x / (1 + np.exp(-x)) ## Swish Activation Function
        return out

    def backward(self, dprev):
        '''
            :param dprev: gradient of next layer:
            :return: downstream gradient
        '''
        # TODO: CHANGE IT
        # Example: derivate of X^2 is 2X
        x = self.x
        dx = dprev * ( (1 / (1 + np.exp(-x))) + (self.x * np.exp(-x) / np.power(1 + np.exp(-x), 2))) ## Swish Activation Backpropogation Function
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


class AffineLayer(LayerWithWeights):
    def __init__(self, input_size, output_size, seed=None):
        super(AffineLayer, self).__init__(input_size, output_size, seed=seed)

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

class Model(Layer):
    def __init__(self, model=None):
        self.layers = model
        self.y = None

    def __call__(self, moduleList):
        for module in moduleList:
            if not isinstance(module, Layer):
                raise TypeError(
                    'All modules in list should be derived from Layer class!')

        self.layers = moduleList

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, y):
        self.y = y.copy()
        dprev = y.copy()
        dprev = self.layers[-1].backward(y)

        for layer in reversed(self.layers[:-1]):
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]

class VanillaSDGOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            if isinstance(m, LayerWithWeights):
                self._optimize(m)

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        
        m.dW += self.reg * m.W # regularization must be put first
        m.W -= self.lr * m.dW ## updating weights
        m.b -= self.lr * m.db ## updating biases
         
        # End of your implementation
       
class SGDWithMomentum(VanillaSDGOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        
        # Save velocities for each model in a dict and use them when needed.
        # Modules can be hashed
        
        self.velocities = {m: 0 for m in model}
        self.biases = {m: 0 for m in model}

    def _optimize(self, m):
        '''
            Optimizer for SGDMomentum
            Do not forget to add L2 regularization!
            :param m: module with weights to optimize
        '''
        # Your implementation starts
        
        m.dW += self.reg * m.W # regularization must be put first (d(W^2) = 2
        
        self.velocities[m] = self.velocities[m]*self.mu - self.lr * m.dW
        m.W += self.velocities[m]
        
        self.biases[m] = self.biases[m] * self.mu - m.db * self.lr
        m.b += self.biases[m]
             
        # End of your implementation
