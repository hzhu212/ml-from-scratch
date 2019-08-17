import numpy as np

from model import Model


class NaiveBayes(Model):
    """Naive Bayes Model"""
    def __init__(self):
        super(NaiveBayes, self).__init__()
        self.labels = None
        self.p_label = None
        self.p_word = None


    def fit(self, x_train, y_train):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        self.labels = np.unique(y_train)

        # plus 1 to smooth the probability (avoid zero-dividing)
        x_train += 1

        p_label = {}
        p_word = {}
        for label in self.labels:
            idx = (y_train == label)
            p_label[label] = np.sum(idx) / len(y_train)
            p_word[label] = np.sum(x_train[idx], axis=0) / np.sum(x_train[idx])

        self.p_label = p_label
        self.p_word = p_word


    def predict1(self, x):
        log_p = [
            np.sum(np.log(self.p_word[label]) * x) + np.log(self.p_label[label])
            for label in self.labels]
        return self.labels[np.argmax(log_p)]


    def predict(self, x_data):
        log_p = np.stack([
                np.sum(np.log(self.p_word[label]) * x_data, axis=1) + np.log(self.p_label[label])
                for label in self.labels],
            axis=1)
        return self.labels[np.argmax(log_p, axis=1)]


if __name__ == '__main__':
    from datasets import email
    import util

    (x_train, y_train), (x_test, y_test) = email.load_data()
    vocabulary = util.get_vocabulary(x_train)
    x_train = np.array([util.word2vector(doc, vocabulary) for doc in x_train])
    x_test = np.array([util.word2vector(doc, vocabulary) for doc in x_test])

    model = NaiveBayes()
    model.fit(x_train, y_train)

    print(y_test)
    print(model.predict(x_test))
    loss, accuracy = model.evaluate(x_test, y_test)
    print('accuracy: {}'.format(accuracy))
