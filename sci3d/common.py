import numpy as np


def get_projection_matrix(near: float,
                          far: float,
                          fov: float,
                          hw_ratio: float,
                          scale_factor: float) -> np.ndarray:
    right = np.tan(fov / 2) * near * scale_factor
    top = right * hw_ratio

    return np.array([
        [near / right, 0, 0, 0],
        [0, near / top, 0, 0],
        [0, 0, -(far + near)/(far - near), -2*far*near / (far - near)],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def inverse_affine(rotation: np.ndarray,
                   translation: np.ndarray):
    assert(rotation.ndim == 2)
    assert(translation.ndim == 2)

    assert(rotation.shape == (3, 3))
    assert(translation.shape == (3, 1))

    inv_rot = rotation.T
    inv_t = inv_rot @ -translation

    result = np.concatenate([inv_rot, inv_t], 1)
    result = np.concatenate([result, np.array([[0, 0, 0, 1]], dtype=np.float32)])

    return result


def forward_affine(rotation: np.ndarray,
                   translation: np.ndarray):
    assert(rotation.ndim == 2)
    assert(translation.ndim == 2)

    assert(rotation.shape == (3, 3))
    assert(translation.shape == (3, 1))

    result = np.concatenate([rotation, translation], 1)
    result = np.concatenate([result, np.array([[0, 0, 0, 1]], dtype=np.float32)])

    return result


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def orthonormalize(matrix: np.ndarray) -> np.ndarray:
    assert(matrix.shape == (3, 3))

    x = normalize_vector(matrix[:, 0:1])

    y = matrix[:, 1:2]
    y = normalize_vector(y - x * np.dot(x.T, y))

    z = np.cross(x.T, y.T).T

    return np.concatenate([
        x, y, z
    ], 1)
