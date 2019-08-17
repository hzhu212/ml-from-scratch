import os
import re

import numpy as np

cur_dir = os.path.dirname(__file__)


def text2token(text):
    text = text.strip().lower()
    tokens = re.split(r'\W+', text)
    tokens = [word for word in tokens if len(word) > 2]
    return tokens


def load_data(test_split=0.2):
    ham_dir = os.path.join(cur_dir, 'ham')
    spam_dir = os.path.join(cur_dir, 'spam')

    ham = []
    for filename in os.listdir(ham_dir):
        with open(os.path.join(ham_dir, filename), 'r', errors='ignore') as f:
            document = text2token(f.read())
            ham.append(document)

    spam = []
    for filename in os.listdir(spam_dir):
        with open(os.path.join(spam_dir, filename), 'r', errors='ignore') as f:
            document = text2token(f.read())
            spam.append(document)

    ham_test_count = int(test_split * len(ham))
    spam_test_count = int(test_split * len(spam))

    x_train = ham[:-ham_test_count] + spam[:-spam_test_count]
    y_train = np.array([0] * (len(ham) - ham_test_count) + [1] * (len(spam) - spam_test_count))
    x_test = ham[-ham_test_count:] + spam[-spam_test_count:]
    y_test = np.array([0] * ham_test_count + [1] * spam_test_count)

    return (x_train, y_train), (x_test, y_test)
