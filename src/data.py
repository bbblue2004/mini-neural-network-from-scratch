import numpy as np
from src.config import SEED, SIGMA, N_PER_CLASS

np.random.seed(SEED)



def gen_train():
    c0 = np.array([-2, -2])
    c1 = np.array([2, 2])
    sigma = SIGMA # standard deviation
    n = N_PER_CLASS # number of points per class
    
    points_0 = c0 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
    points_1 = c1 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
    labels_0 = np.zeros(n)
    labels_1 = np.ones(n)

    X = np.concatenate([points_0, points_1], axis=0)   # np.concatenate(list of arrays, axis)
    y = np.concatenate([labels_0, labels_1], axis=0)

    return X, y


def gen_inference():
    c = np.array([0.0, 0])
    sigma = SIGMA
    n = N_PER_CLASS
    
    points = c + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))

    return points