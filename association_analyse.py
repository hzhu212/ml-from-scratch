import collections
import itertools
import operator

import numpy as np

from model import Model


class Apriori(Model):
    """association analyse with Apriori algorithm"""
    def __init__(self, min_support=0.5, min_confidence=0.5):
        super(Apriori, self).__init__()
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_sets = []
        self.associate_rules = []
        self.supports = []
        self.confidences = []


    def _calc_support(self, aset, dataset):
        """计算一个集合在整个数据集中的支持度"""
        count = sum(aset.issubset(sample) for sample in dataset)
        return float(count) / len(dataset)


    def _calc_confidence(self, antecedent, consequent, dataset):
        """计算一个关联规则的可信度"""
        try:
            return self._calc_support(antecedent | consequent, dataset) / self._calc_support(antecedent, dataset)
        except ZeroDivisionError:
            return 0


    def get_frequent_sets(self, dataset):
        """计算频繁项集"""
        def generate_next_level(prev_sets):
            res = set()
            for i in range(len(prev_sets) - 1):
                for j in range(i + 1, len(prev_sets)):
                    if len(prev_sets[i] - prev_sets[j]) == 1:
                        res.add(prev_sets[i] | prev_sets[j])
            return list(res)

        def filter_sets(sets):
            res = []
            supports = []
            for aset in sets:
                support = self._calc_support(aset, dataset)
                if support >= self.min_support:
                    res.append(aset)
                    supports.append(support)
            return res, supports

        all_items = collections.Counter(itertools.chain.from_iterable(dataset))
        level1, supports1 = filter_sets([frozenset([x]) for x in all_items])
        res = [level1]
        supports = [supports1]
        while True:
            next_sets, next_supports = filter_sets(generate_next_level(res[-1]))
            if len(next_sets) == 0:
                break
            res.append(next_sets)
            supports.append(next_supports)

        return list(itertools.chain.from_iterable(res)), list(itertools.chain.from_iterable(supports))


    def get_associate_rules(self, frequent_sets, dataset):
        """从频繁项集中发掘关联规则"""
        def generate_next_level(prev_rules):
            res = set()
            for i in range(len(prev_rules) - 1):
                for j in range(i + 1, len(prev_rules)):
                    antecedent1, consequent1 = prev_rules[i]
                    antecedent2, consequent2 = prev_rules[j]
                    consequent = consequent1 | consequent2
                    if len(consequent) - len(consequent1) != 1:
                        continue
                    antecedent = (antecedent1 | antecedent2) - consequent
                    if len(antecedent) == 0:
                        continue
                    res.add((antecedent, consequent))
            return list(res)

        def filter_rules(rules):
            res = []
            confs = []
            for antecedent, consequent in rules:
                confidence = self._calc_confidence(antecedent, consequent, dataset)
                if confidence >= self.min_confidence:
                    res.append((antecedent, consequent))
                    confs.append(confidence)
            return res, confs

        def process_one_set(aset):
            if len(aset) <= 1:
                return [], []
            level1, confs1 = filter_rules([(aset - frozenset([x]), frozenset([x])) for x in aset])
            res = [level1]
            confs = [confs1]
            while True:
                next_rules, next_confs = filter_rules(generate_next_level(res[-1]))
                if len(next_rules) == 0:
                    break
                res.append(next_rules)
                confs.append(next_confs)
            return list(itertools.chain.from_iterable(res)), list(itertools.chain.from_iterable(confs))

        res_rules = []
        res_confs = []
        for aset in frequent_sets:
            rules, confs = process_one_set(aset)
            res_rules.append(rules)
            res_confs.append(confs)
        return list(itertools.chain.from_iterable(res_rules)), list(itertools.chain.from_iterable(res_confs))


    def fit(self, dataset):
        # dataset should a list of lists
        dataset = [frozenset(lst) for lst in dataset]
        self.frequent_sets, self.supports = self.get_frequent_sets(dataset)
        self.associate_rules, self.confidences = self.get_associate_rules(self.frequent_sets, dataset)


if __name__ == '__main__':
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

    model = Apriori(min_support=0.5, min_confidence=0.7)
    model.fit(dataset)
    print(model.frequent_sets)
    print(model.supports)
    print(model.associate_rules)
    print(model.confidences)
