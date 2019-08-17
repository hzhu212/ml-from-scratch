import os

import numpy as np

cur_dir = os.path.dirname(__file__)


# http://archive.ics.uci.edu/ml/machine-learning-databases/secom/
def load_data():
    dataset = np.loadtxt(os.path.join(cur_dir, 'secom.txt'))

    # fill NaN with column mean value
    col_mean = np.nanmean(dataset, axis=0)
    idxs = np.where(np.isnan(dataset))
    dataset[idxs] = col_mean[idxs[1]]

    return (dataset, None), (None, None)
