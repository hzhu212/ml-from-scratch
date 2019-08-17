import collections

import numpy as np

from model import Model


def get_shannon_ent(lst):
    counter = collections.Counter(lst)
    ent = 0
    for count in counter.values():
        p = float(count) / len(lst)
        ent += -p * np.log2(p)
    return ent

def split_list(lst):
    """split a list by the value. return indices of each different value."""
    res = collections.defaultdict(list)
    for i, x in enumerate(lst):
        res[x].append(i)
    return dict(res)


class DecisionTree(Model):
    """decision tree.
    Implemented with ID3 algorithm: https://en.wikipedia.org/wiki/ID3_algorithm

    本实现只针对标称型数据，未考虑数值型数据
    """

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.tree = None


    def fit(self, x_train, y_train):
        def build_tree(x_train, y_train, features_left):
            if len(features_left) == 0:
                return collections.Counter(y_train).most_common(1)[0][0]
            if len(np.unique(y_train)) == 1:
                return y_train[0]

            choose_feature_idx = None
            min_ent = np.inf
            for i, ft in enumerate(features_left):
                spliter = split_list(x_train[..., i])
                ent = sum(get_shannon_ent(y_train[idx]) for idx in spliter.values())
                if ent < min_ent:
                    min_ent = ent
                    choose_feature_idx = i
            choose_feature = features_left.pop(choose_feature_idx)
            choose_spliter = split_list(x_train[..., choose_feature_idx])

            col_idx = [True] * x_train.shape[-1]
            col_idx[choose_feature_idx] = False
            x_train = x_train[..., col_idx]

            res = {}
            for feature_val, row_idx in choose_spliter.items():
                res[feature_val] = build_tree(x_train[row_idx], y_train[row_idx], features_left)
            return {choose_feature: res}

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        self.tree = build_tree(x_train, y_train, list(range(x_train.shape[-1])))


    def predict1(self, x):
        def request(x, tree):
            feature = next(iter(tree.keys()))
            answers = tree[feature]
            value = x[feature]
            sub_tree = answers[value]
            if not isinstance(sub_tree, dict):
                return sub_tree
            return request(x, sub_tree)

        return request(x, self.tree)


if __name__ == '__main__':
    x_train = np.array([
        [1, 1],
        [1, 1],
        [1, 0],
        [0, 1],
        [0, 1],
    ])
    y_train = np.array(['yes', 'yes', 'no', 'no', 'no'])

    model = DecisionTree()
    model.fit(x_train, y_train)

    print(model.tree)
    print(y_train)
    print(model.predict(x_train))

    loss, accuracy = model.evaluate(x_train, y_train)
    print('accuracy: {}'.format(accuracy))


    # from datasets import lense

    # (x_train, y_train), (x_test, y_test) = lense.load_data()
    # model = DecisionTree()
    # model.fit(x_train, y_train)
    # loss, accuracy = model.evaluate(x_test, y_test)
    # print('accuracy: {}'.format(accuracy))
