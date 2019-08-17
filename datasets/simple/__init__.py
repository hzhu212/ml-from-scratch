import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data(test_split=0.2):
    dataset = np.loadtxt(os.path.join(cur_dir, 'simple.txt'))
    test_count = int(round(dataset.shape[0] * test_split))
    dataset_train = dataset[:-test_count]
    dataset_test = dataset[-test_count:]

    x_train = dataset_train[:, :2]
    y_train = dataset_train[:, -1].astype(np.int8)
    x_test = dataset_test[:, :2]
    y_test = dataset_test[:, -1].astype(np.int8)

    return (x_train, y_train), (x_test, y_test)
