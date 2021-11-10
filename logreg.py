import numpy as np

def logit(x, w):
    return np.array(np.dot(x, w),dtype=np.float32)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

def generate_batches(X, y, batch_size):
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    for batch_start in range(0, (len(X) // batch_size) * batch_size, batch_size):
        batch_X = X[perm[batch_start: batch_start + batch_size]]
        batch_y = y[perm[batch_start: batch_start + batch_size]]
        yield batch_X, batch_y


class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=20, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)
        losses = []
        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                predictions = self._predict_proba_internal(X_batch)
                loss = self.__loss(y_batch, predictions)
                losses.append(loss)
                self.w -= lr * self.get_grad(X_batch, y_batch, self._predict_proba_internal(X_batch))

        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        grad_basic = X_batch.T @ (predictions - y_batch)
        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def set_weights(self, w_):
        self.w = w_.copy()

    def __loss(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
