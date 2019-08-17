import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data():
    dataset = np.loadtxt(os.path.join(cur_dir, 'data1.txt'))

    return (dataset, None), (None, None)
