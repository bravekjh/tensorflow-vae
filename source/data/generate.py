#!/usr/bin/env python

import pickle
import numpy as np
from numpy.random import *


def sampling(n, mu, sigma):
    return np.reshape(normal(mu, sigma, n), (n, 1))

def corresponding_shuffle(data, target):
    random_indices = permutation(len(data))
    _data = np.zeros_like(data)
    _target = np.zeros_like(target)
    for i, j in enumerate(random_indices):
        _data[i] = data[j]
        _target[i] = target[j]
    return _data, _target

def save_as_pickle(filename, data):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    # config
    n = 500
    x_male = sampling(n, 171.66, 5.60)
    x_female = sampling(n, 158.32, 5.52)
    y_male = np.zeros_like(x_male)
    y_female = np.ones_like(x_male)

    # concat
    data = np.r_[x_male, x_female]
    target = np.r_[y_male, y_female]

    # shuffle
    data, target = corresponding_shuffle(data, target)

    # create dictionary
    dataset = {
        'data': data,
        'target': target
    }

    # save as a pickle file
    save_as_pickle('height.pkl', dataset)
