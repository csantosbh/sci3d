import numpy as np


def look_at(to: np.ndarray,
            up: np.ndarray):
    assert(to.shape == (3, 1))
    assert(up.shape == (3, 1))

    normalize = lambda x: x / np.linalg.norm(x)

    forward = normalize(to)
    up = normalize(up - forward * np.dot(up.T, forward))
    left = np.cross(np.squeeze(up), np.squeeze(forward))[:, np.newaxis]

    return np.concatenate([left, up, forward], 1)
