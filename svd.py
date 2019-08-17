import matplotlib.pyplot as plt
import numpy as np

from model import Model


class SVD(Model):
    """singular value decomposition"""
    def __init__(self, topn=None, top_energy=0.9):
        super(SVD, self).__init__()
        self.topn = topn
        self.top_energy = top_energy
        self.cum_energy_ratio = None
        self.u = None
        self.sigma = None
        self.vh = None
        self.reconstruct_data = None

    def fit(self, dataset):
        dataset = np.asarray(dataset)
        u, sigma, vh = np.linalg.svd(dataset)

        cum_energy_ratio = np.cumsum(sigma ** 2) / np.sum(sigma ** 2)
        if self.topn is None:
            if self.top_energy is None or self.top_energy >= 1.0:
                self.topn = len(sigma)
                self.top_energy = 1.0
            else:
                self.topn = np.searchsorted(cum_energy_ratio, self.top_energy) + 1
                self.top_energy = cum_energy_ratio[self.topn - 1]
        else:
            self.topn = min(self.topn, len(sigma))
            self.top_energy = cum_energy_ratio[self.topn - 1]

        self.cum_energy_ratio = cum_energy_ratio
        self.sigma = sigma
        self.u = u
        self.vh = vh
        self.reconstruct_data = u[:, :self.topn].dot(np.diag(self.sigma[:self.topn])).dot(vh[:self.topn, :])


if __name__ == '__main__':
    from datasets import simple_pca

    (dataset, _), (_, _) = simple_pca.load_data()

    model = SVD(topn=1)
    model.fit(dataset)

    print(model.sigma)
    print(model.topn)
    print(model.top_energy)

    fig, ax = plt.subplots()
    ax.scatter(dataset[:, 0], dataset[:, 1])
    ax.scatter(model.reconstruct_data[:, 0], model.reconstruct_data[:, 1])

    plt.show()
