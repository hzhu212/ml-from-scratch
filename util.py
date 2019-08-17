import matplotlib.pyplot as plt
import numpy as np
import pickle


def scatter_dataset(data, labels, feature_x=0, feature_y=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    data = np.asarray(data)
    labels = np.asarray(labels)
    x = data[:, feature_x]
    y = data[:, feature_y]
    for label in np.unique(labels):
        idx = (labels == label)
        ax.scatter(x[idx], y[idx], label=label)
    ax.legend()
    return ax


def plot_predict(model, x_data, y_data, feature=0, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    x_data = np.asarray(x_data)
    x_plot = x_data[:, feature]
    ax.scatter(x_plot, y_data, s=10)
    y_pred = model.predict(x_data)
    idx = np.argsort(x_plot)
    ax.plot(x_plot[idx], y_pred[idx], 'k-')
    ax.set_xlabel('x[{}]'.format(feature))
    ax.set_ylabel('y')
    if title is not None:
        ax.set_title(title)
    return ax


def min_max_scale(x_data, min_vals=None, max_vals=None):
    """min-max scale a dataset"""
    if min_vals is None:
        min_vals = x_data.min(axis=0)
    if max_vals is None:
        max_vals = x_data.max(axis=0)
    return (x_data - min_vals) / (max_vals - min_vals), min_vals, max_vals


def normalize(x_data, mean=None, std=None):
    if mean is None:
        mean = np.mean(x_data, axis=0)
    if std is None:
        std = np.std(x_data, axis=0)
    return (x_data - mean) / std, mean, std


def get_vocabulary(dataset):
    vocabulary = set()
    for document in dataset:
        vocabulary.update(set(document))
    return sorted(list(vocabulary))


def word2vector(document, vocabulary):
    vector = [0] * len(vocabulary)
    for word in document:
        try:
            idx = vocabulary.index(word)
        except ValueError:
            pass
        else:
            vector[idx] += 1
    return vector


def save(obj, filepath):
    """save python object with pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)
