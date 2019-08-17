import os

import numpy as np

cur_dir = os.path.dirname(__file__)


def load_data():
    dataset_train = np.loadtxt(
        os.path.join(cur_dir, 'train.txt'),
        dtype=[('flying_mileage', 'i4'), ('gaming_time', 'f8'), ('icecream_liter', 'f8'), ('label', 'U10')])
    x_train = np.array(dataset_train[['flying_mileage', 'gaming_time', 'icecream_liter']].tolist())
    y_train = np.array(dataset_train[['label']].tolist()).squeeze()
    mapper = {'didntLike': 1, 'smallDoses': 2, 'largeDoses': 3}
    y_train = np.array(list(map(mapper.get, y_train)))

    dataset_test = np.loadtxt(
        os.path.join(cur_dir, 'test.txt'),
        dtype=[('flying_mileage', 'i4'), ('gaming_time', 'f8'), ('icecream_liter', 'f8'), ('label', 'i1')])
    x_test = np.array(dataset_test[['flying_mileage', 'gaming_time', 'icecream_liter']].tolist())
    y_test = np.array(dataset_test[['label']].tolist()).squeeze()

    return (x_train, y_train), (x_test, y_test)
