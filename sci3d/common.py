from typing import Optional, Tuple
from pathlib import Path

import numpy as np

from nanogui_sci3d import Shader

from sci3d.materials import Material


class BoundingBox(object):
    lower_bound: np.ndarray
    upper_bound: np.ndarray

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @property
    def center(self):
        return (self.lower_bound + self.upper_bound) / 2

    @property
    def width(self):
        return self.upper_bound[0] - self.lower_bound[0]

    @property
    def height(self):
        return self.upper_bound[1] - self.lower_bound[1]

    @property
    def depth(self):
        return self.upper_bound[2] - self.lower_bound[2]

    def union(self, other):
        result = BoundingBox(
            np.minimum(self.lower_bound, other.lower_bound),
            np.maximum(self.upper_bound, other.upper_bound),
        )
        return result


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


def rot_x(radians: float) -> np.ndarray:
    return np.array([
        [1, 0, 0],
        [0, np.cos(radians), -np.sin(radians)],
        [0, np.sin(radians), np.cos(radians)],
    ], dtype=np.float32)


def rot_y(radians: float) -> np.ndarray:
    return np.array([
        [np.cos(radians), 0, np.sin(radians)],
        [0, 1, 0],
        [-np.sin(radians), 0, np.cos(radians)],
    ], dtype=np.float32)


def rot_z(radians: float) -> np.ndarray:
    return np.array([
        [np.cos(radians), -np.sin(radians), 0],
        [np.sin(radians), np.cos(radians), 0],
        [0, 0, 1],
    ], dtype=np.float32)


def cone(radius: float,
         height: float,
         resolution: int = 10) -> Tuple[np.ndarray,
                                        np.ndarray]:
    vertices = np.array([
        [radius * np.cos(k), 0, radius * np.sin(k)]
        for k in np.linspace(0, 2*np.pi, resolution, endpoint=False)
    ] + [
        [0, height, 0],
        [0, 0, 0]
    ], dtype=np.float32)

    top_idx = vertices.shape[0] - 2
    base_center_idx = vertices.shape[0] - 1

    triangles = np.array([
        [(idx + 1) % resolution, idx, top_idx] for idx in range(resolution)
    ] + [
        [idx, (idx + 1) % resolution, base_center_idx] for idx in range(resolution)
    ], dtype=np.uint32)

    return vertices, triangles


class Mesh(object):
    def __init__(self,
                 render_pass, vertices, triangles, normals, colors, projection,
                 ):
        if colors is None:
            colors = np.ones_like(vertices)

        if normals is None:
            vertices, triangles, colors = self._make_triangles_unique(
                vertices, triangles, colors
            )
            normals = self._compute_normals(vertices, triangles)

        self._material = Material(
            render_pass,
            'mesh',
            'mesh_vert.glsl',
            'mesh_frag.glsl'
        )
        self._texture = None

        self._triangle_count = triangles.shape[0]
        self._material.shader.set_buffer('indices', triangles.flatten())
        self._material.shader.set_buffer('position', vertices)
        self._material.shader.set_buffer('projection', projection)
        self._material.shader.set_buffer('normal', normals)
        self._material.shader.set_buffer('color', colors)

        self._bounding_box = BoundingBox(
            np.min(vertices, axis=0),
            np.max(vertices, axis=0)
        )

    def get_bounding_box(self) -> BoundingBox:
        return self._bounding_box

    def get_material(self) -> Material:
        return self._material

    def set_mesh(self,
                 vertices: Optional[np.ndarray],
                 triangles: Optional[np.ndarray],
                 normals: Optional[np.ndarray],
                 colors: Optional[np.ndarray]):
        if triangles is not None:
            self._triangle_count = triangles.shape[0]
            self._material.shader.set_buffer("indices", triangles.flatten())

        if vertices is not None:
            self._material.shader.set_buffer("position", vertices)
            self._bounding_box.lower_bound = np.min(vertices, axis=1)
            self._bounding_box.upper_bound = np.max(vertices, axis=1)

        if normals is not None:
            self._material.shader.set_buffer("normal", normals)

        if colors is not None:
            self._material.shader.set_buffer("color", colors)

    def draw(self,
             world2camera: np.ndarray,
             projection: np.ndarray):
        self._material.shader.set_buffer("world2camera", world2camera.T)
        self._material.shader.set_buffer("projection", projection.T)

        with self._material.shader:
            self._material.shader.draw_array(
                Shader.PrimitiveType.Triangle, 0, self._triangle_count * 3, True)

    def _make_triangles_unique(self, vertices, triangles, colors):
        """
        In some cases, the mesh may be specified in a way that some vertices are
        shared among multiple triangles. This function replicates these vertices
        so that each triangle has its own vertex.
        """
        def make_component_unique(component):
            return np.concatenate([
                np.concatenate([
                    component[tri_vtx:tri_vtx+1, :] for tri_vtx in tri_vtxs
                ], 0)
                for tri_vtxs in triangles
            ], 0)

        vertices = make_component_unique(vertices)
        colors = make_component_unique(colors)

        triangles = np.arange(vertices.shape[0]).reshape((-1, 3)).astype(np.uint32)

        return vertices, triangles, colors

    def _compute_normals(self, vertices, triangles):
        normals = []

        for triangle in triangles:
            v1 = vertices[triangle[1]] - vertices[triangle[0]]
            v2 = vertices[triangle[2]] - vertices[triangle[0]]

            normal = normalize_vector(np.cross(v1, v2))
            for _ in range(3):
                normals.append(normal)

        normals = np.array(normals)

        # Arrange normals in order of vertices as they appear in triangle list
        return normals


class Wireframe(object):
    def __init__(self,
                 render_pass, vertices, indices, colors, projection):
        if colors is None:
            colors = np.ones_like(vertices)

        self._material = Material(
            render_pass,
            'wireframe',
            'wireframe_vert.glsl',
            'wireframe_frag.glsl'
        )
        self._line_count = indices.shape[0]
        self._material.shader.set_buffer('indices', indices.flatten())
        self._material.shader.set_buffer('position', vertices)
        self._material.shader.set_buffer('projection', projection)
        self._material.shader.set_buffer('color', colors)

        self._bounding_box = BoundingBox(
            np.min(vertices, axis=0),
            np.max(vertices, axis=0)
        )

    def get_bounding_box(self) -> BoundingBox:
        return self._bounding_box

    def get_material(self) -> Material:
        return self._material

    def set_mesh(self,
                 indices: Optional[np.ndarray],
                 vertices: Optional[np.ndarray]):
        if indices is not None:
            self._line_count = indices.shape[0]
            self._material.shader.set_buffer("indices", indices.flatten())

        if vertices is not None:
            self._material.shader.set_buffer("position", vertices)
            self._bounding_box.lower_bound = np.min(vertices, axis=1)
            self._bounding_box.upper_bound = np.max(vertices, axis=1)

    def draw(self,
             world2camera: np.ndarray,
             projection: np.ndarray):
        self._material.shader.set_buffer("world2camera", world2camera.T)
        self._material.shader.set_buffer("projection", projection.T)

        with self._material.shader:
            self._material.shader.draw_array(
                Shader.PrimitiveType.Line, 0, self._line_count * 2, True)
