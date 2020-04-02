from .layer import Layer, BatchNorm, Dropout
from .layers_with_weights import LayerWithWeights
import numpy as np

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

        for ix, layer in enumerate(reversed(self.layers[:-1])):
            # print(ix)
            if isinstance(layer, LayerWithWeights):
                dprev = layer.backward(dprev)[0]
            else:
                dprev = layer.backward(dprev)
        return dprev

    def train(self):
        ''' Toggle train mode on
            Only affects Dropout and Batch Normalization
        '''
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                layer.mode = 'train'

    def test(self):
        ''' Toggle train mode on
            Only affects Dropout and Batch Normalization
        '''
        for layer in self.layers:
            if isinstance(layer, (Dropout, BatchNorm)):
                layer.mode = 'test'

    def __iter__(self):
        return iter(self.layers)

    def __repr__(self):
        return 'Model consisting of {}'.format('/n -- /t'.join(self.layers))

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, i):
        return self.layers[i]
