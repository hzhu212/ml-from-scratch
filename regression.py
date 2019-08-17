import matplotlib.pyplot as plt
import numpy as np

from model import Model


def mse(y_hat, y):
    return np.mean(np.power(y_hat - y, 2))


def pearson_distance(y_hat, y):
    return 1 - np.corrcoef(y_hat, y)[0, 1]


class LinearRegression(Model):
    """standard linear regression"""
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.loss_fun = pearson_distance
        self.w = None

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        m, n = x_train.shape
        x_train = np.concatenate([np.ones((m, 1)), x_train], axis=1)
        self.w = np.linalg.inv(np.dot(x_train.T, x_train)).dot(x_train.T).dot(y_train)

    def predict(self, x_data):
        x_data = np.asarray(x_data)
        m, n = x_data.shape
        x_data = np.concatenate([np.ones((m, 1)), x_data], axis=1)
        y_pred = np.dot(x_data, self.w)
        return y_pred


class RidgeRegression(LinearRegression):
    """ridge regression
    equals to standard linear regression when the ridge is zero.

    岭回归有两个作用：
    1. 当样本数低于特征数时，仍可以进行回归，不会产生矩阵不可逆的情况。
    2. 相当于以权重的二范数作为正则项，能够精简权重数量，权重值越大，代表对应的特征越重要
    """
    def __init__(self, ridge=0):
        super(RidgeRegression, self).__init__()
        self.loss_fun = pearson_distance
        self.w = None
        self.ridge = ridge

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        m, n = x_train.shape
        x_train = np.concatenate([np.ones((m, 1)), x_train], axis=1)
        self.w = np.linalg.inv(np.dot(x_train.T, x_train) + np.eye(n + 1) * self.ridge).dot(x_train.T).dot(y_train)


class LocalWeightedLinearRegression(Model):
    """local weighted linear regression"""
    def __init__(self, k=1.0):
        super(LocalWeightedLinearRegression, self).__init__()
        self.loss_fun = pearson_distance
        self.k = k

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        m, n = x_train.shape
        x_train = np.concatenate([np.ones((m, 1)), x_train], axis=1)
        self.x_train = x_train
        self.y_train = y_train

    def predict1(self, x):
        x = np.concatenate([[1], x], axis=0)
        m, n = self.x_train.shape
        weights = np.exp(np.power(x - self.x_train, 2).sum(axis=1) / (-2 * self.k**2))
        weights = np.diag(weights)
        w = np.linalg.inv(self.x_train.T.dot(weights).dot(self.x_train)).dot(self.x_train.T).dot(weights).dot(self.y_train)
        return x.dot(w)


if __name__ == '__main__':
    from datasets import simple_regression
    import util

    (x_train, y_train), (x_test, y_test) = simple_regression.load_data()
    x_train, mean, std = util.normalize(x_train)
    x_test, _, _ = util.normalize(x_test, mean, std)

    model_std = LinearRegression()
    model_std.fit(x_train, y_train)
    util.plot_predict(model_std, x_train, y_train, title='Standard Linear Regression')
    loss, accuracy = model_std.evaluate(x_train, y_train)
    print('loss: {}'.format(loss))

    model_ridge = RidgeRegression(ridge=0)
    model_ridge.fit(x_train, y_train)
    util.plot_predict(model_ridge, x_train, y_train, title='Ridge Regression (ridge = {})'.format(model_ridge.ridge))
    loss, accuracy = model_ridge.evaluate(x_train, y_train)
    print('loss: {}'.format(loss))

    model_lwlr = LocalWeightedLinearRegression()
    model_lwlr.fit(x_train, y_train)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle('Local Weighted Linear Regression')
    for i, k in enumerate([1, 0.1, 0.01]):
        model_lwlr.k = k
        util.plot_predict(model_lwlr, x_train, y_train, ax=axes[i], title='k = {}'.format(k))
        loss, accuracy = model_lwlr.evaluate(x_train, y_train)
        print('loss: {}'.format(loss))

    plt.show()
