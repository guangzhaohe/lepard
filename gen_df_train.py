from typing import List, Dict

import numpy as np


def procrustes_alignment(X, Y):
    assert X.shape == Y.shape, "Point clouds must have the same shape"
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)
    X0 = X - mu_X
    Y0 = Y - mu_Y
    H = Y0.T @ X0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_X - R @ mu_Y
    Y_aligned = (R @ Y.T).T + t
    return Y_aligned, R, t


def generate_training_labels(data: Dict):
    pass


if __name__ == '__main__':
    pass    
