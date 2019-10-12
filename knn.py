import collections
import heapq

import numpy as np

from model import Model


class KNN(Model):
    """k-nearest neighbour model"""

    def __init__(self, k):
        super(KNN, self).__init__()
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

    def predict1(self, x):
        dist = np.sum(np.power(self.x_train - x, 2), axis=1)
        idx = heapq.nsmallest(self.k, range(len(dist)), key=dist.__getitem__)
        labels = self.y_train.squeeze()[idx]
        y = collections.Counter(labels).most_common(1)[0][0]
        return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from datasets import dating
    import util

    (x_train, y_train), (x_test, y_test) = dating.load_data()
    util.scatter_dataset(x_train, y_train)
    plt.show()
    x_train, mean, std = util.normalize(x_train)
    x_test, _, _ = util.normalize(x_test, mean, std)

    model = KNN(k=3)
    model.fit(x_train, y_train)

    print(y_test[:10])
    print(model.predict(x_test)[:10])
    loss, accuracy = model.evaluate(x_test, y_test)
    print('accuracy: {}'.format(accuracy))

