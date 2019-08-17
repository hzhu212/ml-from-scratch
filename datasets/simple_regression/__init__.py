import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data(test_split=0.1):
    dataset = np.loadtxt(os.path.join(cur_dir, 'data1.txt'))
    test_count = int(round(dataset.shape[0] * test_split))
    data_train = dataset[:-test_count]
    data_test = dataset[-test_count:]

    x_train = data_train[:, :-1]
    y_train = data_train[:, -1]
    x_test = data_test[:, :-1]
    y_test = data_test[:, -1]

    return (x_train, y_train), (x_test, y_test)
