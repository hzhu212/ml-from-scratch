import numpy as np

def load_data():
    x_train = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1],
        ])
    y_train = np.array(['A', 'A', 'B', 'B'])

    x_test = np.array([
        [0.8, 0.9],
        [0.1, 0.2],
        ])
    y_test = np.array(['A', 'B'])

    return (x_train, y_train), (x_test, y_test)
