from .layers_with_weights import LayerWithWeights


class VanillaSDGOptimizer(object):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4):
        self.reg = regularization_str
        self.model = model
        self.lr = lr

    def optimize(self):
        for m in self.model:
            self._optimize(m)

    def _optimize(self, m):
        if isinstance(m, LayerWithWeights):
            m.W += -(m.dW * self.lr + m.W * self.reg)
            m.b += -m.db * self.lr


class SGDWithMomentum(VanillaSDGOptimizer):
    def __init__(self, model, lr=1e-3, regularization_str=1e-4, mu=.5):
        self.reg = regularization_str
        self.model = model
        self.lr = lr
        self.mu = mu
        self.velocities = {m: 0 for m in model}

    def _optimize(self, m):
        if isinstance(m, LayerWithWeights):
            v = self.velocities[m]
            v = self.mu*v - self.lr*m.dW
            m.W += v - m.W * self.reg
            m.b += -m.db * self.lr
            self.velocities[m] = v
