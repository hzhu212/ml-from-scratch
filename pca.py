import matplotlib.pyplot as plt
import numpy as np

from model import Model


class PCA(Model):
    """primary composition analyse"""
    def __init__(self, topn=None, top_energy=0.9):
        super(PCA, self).__init__()
        self.topn = topn
        self.top_energy = top_energy
        self.eig_values = None
        self.eig_vectors = None
        self.cum_energy_ratio = None
        self.low_data = None
        self.reconstruct_data = None

    def fit(self, dataset):
        dataset = np.array(dataset)
        mean = np.mean(dataset, axis=0)
        dataset -= mean

        cov_mat = np.cov(dataset, rowvar=False)
        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        idx = np.argsort(eig_values)[::-1]
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx]

        cum_energy_ratio = np.cumsum(eig_values) / np.sum(eig_values)
        if self.topn is None:
            if self.top_energy is None or self.top_energy >= 1.0:
                self.topn = len(eig_values)
                self.top_energy = 1.0
            else:
                self.topn = np.searchsorted(cum_energy_ratio, self.top_energy) + 1
                self.top_energy = cum_energy_ratio[self.topn - 1]
        else:
            self.topn = min(self.topn, len(eig_values))
            self.top_energy = cum_energy_ratio[self.topn - 1]

        self.eig_values = eig_values
        self.eig_vectors = eig_vectors
        self.cum_energy_ratio = cum_energy_ratio

        self.low_data = np.dot(dataset, self.eig_vectors[:, :self.topn])
        self.reconstruct_data = np.dot(self.low_data, self.eig_vectors[:, :self.topn].T) + mean


if __name__ == '__main__':
    from datasets import simple_pca

    (dataset, _), (_, _) = simple_pca.load_data()

    model = PCA(topn=1)
    model.fit(dataset)

    print('eig_values:', model.eig_values)
    print('cum_energy_ratio:', model.cum_energy_ratio)
    print('topn:', model.topn)
    print('top_energy:', model.top_energy)

    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1])
    # ax.scatter(model.low_data, np.zeros(model.low_data.shape))
    ax.scatter(model.reconstruct_data[:, 0], model.reconstruct_data[:, 1])

    plt.show()


    from datasets import secom

    (dataset, _), (_, _) = secom.load_data()
    print('original dataset shape:', dataset.shape)

    model = PCA(top_energy=0.95)
    model.fit(dataset)

    print('eig_values:', model.eig_values[:30])
    print('cum_energy_ratio:', model.cum_energy_ratio[:30])
    print('topn:', model.topn)
    print('top_energy:', model.top_energy)

    fig, ax = plt.subplots()
    ax.plot(model.cum_energy_ratio[:30], marker='o')
    ax.set_title('accumulative ratio of eigenvalues')

    print('compressed dataset shape:', model.low_data.shape)
    plt.show()
