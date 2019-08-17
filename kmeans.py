import matplotlib.pyplot as plt
import numpy as np

from model import Model


"""Distance functions.
these functions should be able to handle multiple times multiple distances.
Assume the input is vec1(m1*n) and vec2(m2*n), the output should be like D(m1*m2),
in which D[i,j] represents the distance between vec1[i] and vec2[j].
"""

def euler_distance(vec1, vec2):
    vec1 = np.atleast_2d(vec1)
    vec2 = np.atleast_2d(vec2)
    assert vec1.shape[1] == vec2.shape[1]
    m1, n = vec1.shape
    m2, n = vec2.shape
    vec1 = vec1.reshape((m1, n, 1))
    vec2 = vec2.T.reshape((1, n, m2))
    return np.sqrt(np.power(vec1 - vec2, 2).sum(axis=1))


def cos_distance(vec1, vec2):
    vec1 = np.atleast_2d(vec1)
    vec2 = np.atleast_2d(vec2)
    assert vec1.shape[1] == vec2.shape[1]
    m1, n = vec1.shape
    m2, n = vec2.shape
    vec1 = np.tile(vec1.reshape((m1, n, 1)), (1, 1, m2))
    vec2 = np.tile(vec2.T.reshape((1, n, m2)), (m1, 1, 1))
    return 1 - (vec1 * vec2).sum(axis=1) / (np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1))


def pearson_distance(vec1, vec2):
    vec1 = np.atleast_2d(vec1)
    vec2 = np.atleast_2d(vec2)
    assert vec1.shape[1] == vec2.shape[1]
    m1, n = vec1.shape
    m2, n = vec2.shape
    res = []
    for i in range(m2):
        res.append(np.corrcoef(vec2[i], vec1)[0, 1:])
    return 1 - np.stack(res).T


def plot_cluster(dataset, centers, assignment, feature_x=0, feature_y=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    dataset = np.asarray(dataset)
    assignment = np.asarray(assignment)
    colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    for i in np.unique(assignment):
        idx = (assignment == i)
        ax.scatter(dataset[idx, feature_x], dataset[idx, feature_y], label=i, c=colors[i%len(colors)])
        ax.scatter(centers[i, feature_x], centers[i, feature_y], s=200, marker='+', c=colors[i%len(colors)])
    ax.legend()
    return ax


class KMeans(Model):
    """K-Means model"""
    def __init__(self, k, max_iter=100, distance_fun=euler_distance):
        super(KMeans, self).__init__()
        self.k = k
        self.max_iter = max_iter
        self.distance_fun = distance_fun
        self.centers = None
        self.assignment = None
        self.loss = None

    def fit(self, x_train):
        x_train = np.asarray(x_train)
        m, n = x_train.shape
        mins = np.min(x_train, axis=0)
        maxs = np.max(x_train, axis=0)
        # init center points with random values in range
        self.centers = np.random.random((self.k, n)) * (maxs - mins) + mins
        self.assignment = np.zeros(m, dtype=np.int32)

        history = []
        for _ in range(self.max_iter):
            # calculate distances and assign cluster for all samples
            distance = self.distance_fun(x_train, self.centers)
            assignment = np.argmin(distance, axis=1)
            self.loss = np.array([np.sum(distance[assignment == i, i]) for i in range(self.k)])
            history.append((self.centers.copy(), assignment.copy(), self.loss.copy()))

            changed = np.any(assignment != self.assignment)
            if not changed:
                break
            self.assignment = assignment

            # update centers
            for i in range(self.k):
                idx = (assignment == i)
                if np.sum(idx) == 0:
                    continue
                self.centers[i] = np.mean(x_train[idx], axis=0)
        return history


class BinaryKMeans(Model):
    """Binary K-Means model"""
    def __init__(self, k, max_iter=100, distance_fun=euler_distance):
        super(BinaryKMeans, self).__init__()
        self.k = k
        self.max_iter = max_iter
        self.distance_fun = distance_fun
        self.centers = None
        self.assignment = None
        self.loss = None

    def fit(self, x_train):
        x_train = np.asarray(x_train)
        m, n = x_train.shape
        mins = np.min(x_train, axis=0)
        maxs = np.max(x_train, axis=0)
        # init center points with random values in range
        self.centers = np.random.random((1, n)) * (maxs - mins) + mins
        self.assignment = np.zeros(m, dtype=np.int32)
        self.loss = np.sum(self.distance_fun(x_train, self.centers), axis=0)

        history = [(self.centers.copy(), self.assignment.copy(), self.loss.copy())]
        while self.centers.shape[0] < self.k:
            max_loss_decrease = 0
            split_idx = 0
            split_model = None
            # for each existed cluster, try to split it and find the best one to split
            for i in range(self.centers.shape[0]):
                model = KMeans(k=2, max_iter=self.max_iter, distance_fun=self.distance_fun)
                model.fit(x_train[self.assignment == i])
                new_loss = np.sum(model.loss)
                loss_decrease = self.loss[i] - new_loss
                if loss_decrease > max_loss_decrease:
                    max_loss_decrease = loss_decrease
                    split_idx = i
                    split_model = model
            self.centers[split_idx] = split_model.centers[0]
            self.centers = np.concatenate([self.centers, split_model.centers[1][np.newaxis]], axis=0)
            self.loss[split_idx] = split_model.loss[0]
            self.loss = np.concatenate([self.loss, [split_model.loss[1]]])
            reassign_idx = np.where(self.assignment == split_idx)[0]
            self.assignment[reassign_idx[split_model.assignment == 1]] = self.centers.shape[0] - 1

            history.append((self.centers.copy(), self.assignment.copy(), self.loss.copy()))
        return history


if __name__ == '__main__':
    from matplotlib.animation import FuncAnimation
    from datasets import simple_cluster
    import util

    (x_train, _), (_, _) = simple_cluster.load_data()
    x_train, mean, std = util.normalize(x_train)

    # model = KMeans(k=4, max_iter=50, distance_fun=euler_distance)
    model = BinaryKMeans(k=4, max_iter=50, distance_fun=euler_distance)
    history = model.fit(x_train)

    fig, ax = plt.subplots()
    def update(frame):
        centers, assignment, loss = frame
        ax.clear()
        _ = plot_cluster(x_train, centers, assignment, ax=ax)
        return [_]
    ani = FuncAnimation(fig, update, frames=history, init_func=None, interval=500, repeat=False, blit=True)

    plt.show()
