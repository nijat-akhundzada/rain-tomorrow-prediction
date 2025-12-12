import numpy as np


class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        y_ = np.where(y <= 0, -1, 1)

        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

        return self

    def decision_function(self, X):
        """Returns the raw decision scores (distance from hyperplane)."""
        X = np.array(X)
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        approx = self.decision_function(X)
        return np.where(np.sign(approx) == -1, 0, 1)

    def predict_proba(self, X):
        # Required for soft voting
        scores = self.decision_function(X)
        prob_positive = 1.0 / (1.0 + np.exp(-scores))
        prob_negative = 1.0 - prob_positive
        return np.column_stack([prob_negative, prob_positive])
