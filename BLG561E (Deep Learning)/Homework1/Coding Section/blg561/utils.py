import numpy as np


def load_mnist_data():
    X, y = [], []
    with open('dataset/train.csv') as f:
        data = [line.rstrip('\n').split(',') for line in f]
        data = np.array(data)

    X = data[:, 1:]
    y = data[:, 0]

    def normalize(X):
        X_mean = X.mean()
        X_std = X.std()
        return (X - X_mean) / X_std
    
    return normalize(X), y
