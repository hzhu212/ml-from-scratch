import numpy as np


class Model(object):
    """base Model class"""
    def __init__(self, *args, **kw):
        pass

    def fit(self, x_train, y_train):
        """train model"""
        raise NotImplementedError('{}.fit is not implemented'.format(self.__class__.__name__))

    def predict1(self, x):
        """predict on one sample"""
        raise NotImplementedError('{}.predict1 is not implemented'.format(self.__class__.__name__))

    def predict(self, x_data):
        """predict on dataset.
        by default, if predict1 has been implemented, just predict samples one by one.
        """
        y_pred = []
        for x in x_data:
            try:
                yp = self.predict1(x)
            except NotImplementedError:
                raise NotImplementedError('{}.predict is not implemented'.format(self.__class__.__name__))
            y_pred.append(yp)
        return np.array(y_pred)

    def forward(self, x_data):
        """use predict as the default forward calculation function"""
        return self.predict(x_data)

    def evaluate(self, x_test, y_test):
        """evaluate on test dataset.
        return a tuple containing loss and accuracy.
        """
        loss = None
        if hasattr(self, 'loss_fun') and callable(self.loss_fun):
            y_hat = self.forward(x_test)
            loss = self.loss_fun(y_hat, y_test)

        y_pred = self.predict(x_test)
        true_count = sum(yp == y for yp, y in zip(y_pred, y_test))
        accuracy = float(true_count) / len(y_test)
        return loss, accuracy
