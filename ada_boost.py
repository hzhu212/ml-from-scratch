import numpy as np

from model import Model


class AdaBoost(Model):
    """AdaBoost model"""
    def __init__(self, max_model):
        super(AdaBoost, self).__init__()
        self.max_model = max_model
        self.models = None
        self.alphas = None

    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        m, n = x_train.shape
        sample_weight = np.ones(y_train.shape) / m

        models = []
        alphas = []
        total_pred = 0
        for i in range(self.max_model):
            model = Stump()
            model.fit(x_train, y_train, sample_weight)
            error = model.params['error']
            alpha = 0.5 * np.log((1 - error) / error)

            y_pred = model.predict(x_train)
            correct = (y_pred == y_train)
            sample_weight = sample_weight * np.exp(-1 * np.sign(correct - 0.5) * alpha)
            sample_weight = sample_weight / np.sum(sample_weight)

            models.append(model)
            alphas.append(alpha)
            total_pred += alpha * y_pred

            total_wrong = (np.sign(total_pred) != y_train)
            total_error = sum(total_wrong) / m
            # print('total error: {}'.format(total_error))
            if total_error == 0:
                break
        self.models = models
        self.alphas = alphas

    def forward(self, x_data):
        x_data = np.asarray(x_data)
        m, n = x_data.shape
        total_pred = 0
        for model, alpha in zip(self.models, self.alphas):
            total_pred += alpha * model.predict(x_data)
        return total_pred

    def predict(self, x_data):
        total_pred = self.forward(x_data)
        return np.sign(total_pred)


class Stump(Model):
    """one layer data-weighted decision tree"""
    def __init__(self):
        super(Stump, self).__init__()
        self.params = None

    def fit(self, x_train, y_train, sample_weight=None):
        N_SPLIT = 10
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        m, n = x_train.shape
        if sample_weight is None:
            sample_weight = np.ones(y_train.shape) / m
        else:
            sample_weight = np.asarray(sample_weight)

        min_error = np.inf
        params = None
        for feature in range(n):
            left = x_train[:, feature].min()
            right = x_train[:, feature].max()
            step = (right - left) / N_SPLIT
            for i in range(-1, N_SPLIT + 1):
                split_at = left + i * step
                for reverse in (False, True):
                    y_pred = np.sign((x_train[:, feature] > split_at) - 0.5)
                    if reverse:
                        y_pred *= -1
                    wrong = (y_pred != y_train)
                    weighted_error = np.dot(wrong, sample_weight)
                    # print(sample_weight.shape)
                    if weighted_error < min_error:
                        min_error = weighted_error
                        params = {'feature': feature, 'split_at': split_at, 'reverse': reverse, 'error': weighted_error}
        self.params = params

    def predict(self, x_data):
        def _predict(x_data, feature, split_at, reverse, **kw):
            y_pred = (x_data[:, feature] > split_at).astype(np.int8)
            y_pred[y_pred == 0] = -1
            if reverse:
                y_pred *= -1
            return y_pred
        return _predict(x_data, **self.params)


if __name__ == '__main__':
    from datasets import horse_colic

    (x_train, y_train), (x_test, y_test) = horse_colic.load_data()
    y_train = np.sign(y_train - 0.5)
    y_test = np.sign(y_test - 0.5)

    model = AdaBoost(max_model=50)
    model.fit(x_train, y_train)
    loss, accuracy = model.evaluate(x_train, y_train)
    print('trianing accuracy: {}'.format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test)
    print('testing accuracy: {}'.format(accuracy))
