import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data():
    def loadtxt(filepath):
        with open(filepath, 'r') as f:
            arr = np.loadtxt(f)
        x = arr[:, :-1]
        y = arr[:, -1].astype(np.int32)
        return x, y

    x_train, y_train = loadtxt(os.path.join(cur_dir, 'train.txt'))
    x_test, y_test = loadtxt(os.path.join(cur_dir, 'test.txt'))

    return (x_train, y_train), (x_test, y_test)
