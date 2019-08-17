import numpy as np

from model import Model


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def binary_cross_entropy(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


class Logistic(Model):
    """Logistic Model"""
    def __init__(self):
        super(Logistic, self).__init__()
        self.w = None


    def loss_fun(self, y_hat, y):
        """use cross entropy as loss function"""
        return np.mean(binary_cross_entropy(y_hat, y))


    def fit(self, x_train, y_train, lr=0.01, batch_size=16, epoch=20, validation_split=0.1):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        y_train = y_train.reshape((-1, 1))

        m, n = x_train.shape
        self.w = np.random.random((n + 1, 1)) * 0.01

        validation_count = int(round(m * validation_split))
        x_train = x_train[:-validation_count]
        y_train = y_train[:-validation_count]
        x_validation = x_train[-validation_count:]
        y_validation = y_train[-validation_count:]

        indices = list(range(x_train.shape[0]))
        history = []
        for i in range(epoch):
            np.random.shuffle(indices)
            for j in range(int(np.ceil(x_train.shape[0] / batch_size))):
                idx = indices[j*batch_size:(j+1)*batch_size]
                y_hat = self.forward(x_train[idx])
                padded = np.concatenate([np.ones((len(idx), 1)), x_train[idx]], axis=1)
                grad = np.dot(padded.transpose(), y_hat - y_train[idx])
                self.w -= lr * grad

            loss, accuracy = self.evaluate(x_validation, y_validation)
            history.append({'loss': loss, 'accuracy': accuracy, 'weight': self.w.copy()})
        return history


    def forward(self, x_data):
        # add one column for b
        m, n = x_data.shape
        x_data = np.concatenate([np.ones((m, 1)), x_data], axis=1)
        assert n + 1 == self.w.shape[0]

        y_hat = sigmoid(np.dot(x_data, self.w))
        return y_hat


    def predict(self, x_data):
        y = self.forward(x_data)
        return (y > 0.5).astype(np.int8)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from datasets import simple
    import util

    (x_train, y_train), (x_test, y_test) = simple.load_data()
    x_train, mean, std = util.normalize(x_train)
    x_test, _, _ = util.normalize(x_test, mean, std)

    model = Logistic()
    history = model.fit(x_train, y_train, lr=0.01, batch_size=16, epoch=10, validation_split=0.15)

    # plot training progress
    fig1, ax1 = plt.subplots()
    util.scatter_dataset(x_train, y_train, ax=ax1)
    weight = [info['weight'] for info in history[::2]]
    x = [x_train[:, 0].min(), x_train[:, 0].max()]
    for i, w in enumerate(weight):
        line = lambda x: -w[1]/w[2] * x - w[0]/w[2]
        ax1.plot(x, [line(x[0]), line(x[1])], 'k-', alpha=1-(i+1)/len(weight))
    ax1.set_xlabel('x[0]')
    ax1.set_ylabel('x[1]')

    fig2, ax2 = plt.subplots()
    loss = [info['loss'] for info in history]
    accuracy = [info['accuracy'] for info in history]
    ax2.plot(loss, label='loss')
    ax2.plot(accuracy, label='accuracy')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Validation Loss and Accuracy')
    plt.show()

    loss, accuracy = model.evaluate(x_test, y_test)
    print('loss: {}, accuracy: {}'.format(loss, accuracy))
