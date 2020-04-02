
import numpy as np

def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def grad_check(f, x, df=None, h=1e-5):
    grad = np.zeros(x.shape)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        plus = f(x).copy()
        x[ix] = oldval - h
        minus = f(x).copy()
        x[ix] = oldval
        if df is not None:
            grad[ix] = np.sum((plus - minus) * df) / (2 * h)
        else:
            grad[ix] = (plus - minus) / (2 * h)
        it.iternext()
    return grad

