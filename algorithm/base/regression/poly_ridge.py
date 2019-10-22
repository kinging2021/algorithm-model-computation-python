import numpy as np
from sklearn.linear_model import Ridge


def get_poly(x, times):
    x = np.array(x)
    x = x.reshape(-1)
    X = []
    for i in range(times):
        X.append(x ** i)

    X = np.array(X).T
    return X


class PolyRidge:
    def __init__(self, alpha=1, times=8):
        self.alpha = alpha
        self.times = times
        self.model = Ridge(alpha=self.alpha)
        return

    def fit(self, data, y):
        X = get_poly(data, self.times)
        self.model.fit(X, y)
        return

    def predict(self, data):
        X = get_poly(data, self.times)
        return self.model.predict(X)
