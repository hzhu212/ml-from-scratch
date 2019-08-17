import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data(test_split=0.2):
    file = os.path.join(cur_dir, 'lense.txt')
    x = []
    y = []
    with open(file, 'r') as f:
        for line in f:
            row = line.strip().split('\t')
            x.append(row[:-1])
            y.append(row[-1])

    test_count = int(len(y) * test_split)
    x_train = np.array(x)
    y_train = np.array(y)
    x_test = np.array(x[-test_count:])
    y_test = np.array(y[-test_count:])

    return (x_train, y_train), (x_test, y_test)
