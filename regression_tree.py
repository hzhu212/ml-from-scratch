import matplotlib.pyplot as plt
import numpy as np

from model import Model
from regression import LinearRegression


def linear_solve(x_data, y_data):
    model = LinearRegression()
    model.fit(x_data, y_data)
    return model.predict(x_data)


def square_error(y_hat, y):
    y_hat = np.asarray(y_hat)
    return np.sum(np.power(y_hat - y, 2))


class TreeNode(object):
    """tree node"""
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.model = None
        self.left = None
        self.right = None


class RegressionTree(Model):
    """tree regression based on CART algorithm"""
    def __init__(self, min_sample_count=1, min_error_decrease=0):
        super(RegressionTree, self).__init__()
        self.min_sample_count = min_sample_count
        self.min_error_decrease = min_error_decrease
        self.head = None

    def fit(self, x_train, y_train):
        def best_split(x_data, y_data):
            m, n = x_data.shape
            origin_error = square_error(linear_solve(x_data, y_data), y_data)
            min_error = np.inf
            min_feature = None
            split_val = None
            for feature in range(n):
                for value in set(x_data[:, feature]):
                    idx = (x_data[:, feature] <= value)
                    if np.sum(idx) < self.min_sample_count or np.sum(~idx) < self.min_sample_count:
                        continue
                    error_left = square_error(linear_solve(x_data[idx], y_data[idx]), y_data[idx])
                    error_right = square_error(linear_solve(x_data[~idx], y_data[~idx]), y_data[~idx])
                    error = error_left + error_right
                    if error < min_error:
                        min_error = error
                        min_feature = feature
                        split_val = value
            if origin_error - min_error < self.min_error_decrease:
                min_feature, split_val = None, None
            return min_feature, split_val

        def build_tree(x_data, y_data):
            feature, split_val = best_split(x_data, y_data)
            node = TreeNode(feature, split_val)
            if feature is None:
                node.model = LinearRegression()
                node.model.fit(x_data, y_data)
            else:
                idx = (x_data[:, feature] <= split_val)
                node.left = build_tree(x_data[idx], y_data[idx])
                node.right = build_tree(x_data[~idx], y_data[~idx])
            return node

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        self.head = build_tree(x_train, y_train)

    def predict1(self, x):
        x = np.asarray(x)
        node = self.head
        while node.feature is not None:
            if x[node.feature] <= node.value:
                node = node.left
            else:
                node = node.right
        return node.model.predict(x.reshape((1, -1)))[0]


if __name__ == '__main__':
    from datasets import simple_regression
    import util

    (x_train, y_train), (x_test, y_test) = simple_regression.load_data()
    x_train, mean, std = util.normalize(x_train)
    x_test, _, _ = util.normalize(x_test, mean, std)

    model = RegressionTree(min_sample_count=10, min_error_decrease=0)
    model.fit(x_train, y_train)
    util.plot_predict(model, x_train, y_train)

    plt.show()
