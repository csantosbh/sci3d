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


def sphere(radius: float):
    num_samples = 20
    vrange = np.mgrid[0:np.pi:num_samples*1j]
    hrange = np.mgrid[-np.pi:np.pi:num_samples*2j]

    vertices = []
    indices = []
    normals = []
    uvs = []

    for y0, y1 in zip(vrange[:-1], vrange[1:]):
        for x0, x1 in zip(hrange[:-1], hrange[1:]):
            v0 = np.array([
                np.sin(x0) * np.sin(y0), np.cos(y0), np.cos(x0) * np.sin(y0)
            ])
            v1 = np.array([
                np.sin(x0) * np.sin(y1), np.cos(y1), np.cos(x0) * np.sin(y1)
            ])
            v2 = np.array([
                np.sin(x1) * np.sin(y1), np.cos(y1), np.cos(x1) * np.sin(y1)
            ])
            v3 = np.array([
                np.sin(x1) * np.sin(y0), np.cos(y0), np.cos(x1) * np.sin(y0)
            ])

            uv0 = np.array([x0 * 0.5 / np.pi + 0.5, y0 / np.pi])
            uv1 = np.array([x0 * 0.5 / np.pi + 0.5, y1 / np.pi])
            uv2 = np.array([x1 * 0.5 / np.pi + 0.5, y1 / np.pi])
            uv3 = np.array([x1 * 0.5 / np.pi + 0.5, y0 / np.pi])

            base_idx = len(vertices)

            vertices += [v0, v1, v2, v3]
            normals += [v0, v1, v2, v3]
            uvs += [uv0, uv1, uv2, uv3]

            indices.append([base_idx + 0, base_idx + 1, base_idx + 2])
            indices.append([base_idx + 0, base_idx + 2, base_idx + 3])

    vertices = np.array(vertices, dtype=np.float32) * radius
    indices = np.array(indices, dtype=np.uint32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)

    return vertices, indices, normals, uvs


def sphere2(radius: float) -> Tuple[np.ndarray,
                                   np.ndarray,
                                   np.ndarray]:

    v0 = [
        [radius, 0, 0],   # 0
        [-radius, 0, 0],  # 1
        [0, radius, 0],   # 2
        [0, -radius, 0],  # 3
        [0, 0, radius],   # 4
        [0, 0, -radius],  # 5
    ]
    t0 = [
        [2, 4, 0],
        [4, 2, 1],
        [5, 2, 0],
        [2, 5, 1],

        [4, 3, 0],
        [3, 4, 1],
        [3, 5, 0],
        [5, 3, 1],
    ]

    vtx_idx = dict()

    def get_vtx_idx(v, vertices):
        v = tuple(v)
        if v not in vtx_idx:
            vtx_idx[v] = len(vertices)
            vertices.append(v)
        return v, vtx_idx[v]

    def recurse_sphere(vertices,
                       triangles,
                       depth=4):
        refined_triangles = []

        def take_mid_point(indices):
            mid_point = np.mean(np.take(vertices, indices, axis=0), 0)
            mid_point = mid_point / np.linalg.norm(mid_point)
            return mid_point

        for triangle in triangles:
            a, a_idx = get_vtx_idx(take_mid_point([triangle[0], triangle[1]]), vertices)
            b, b_idx = get_vtx_idx(take_mid_point([triangle[0], triangle[2]]), vertices)
            c, c_idx = get_vtx_idx(take_mid_point([triangle[1], triangle[2]]), vertices)

            refined_triangles += [
                [a_idx, b_idx, triangle[0]],
                [c_idx, a_idx, triangle[1]],
                [b_idx, c_idx, triangle[2]],
                [b_idx, a_idx, c_idx]
            ]

        triangles = np.array(refined_triangles, dtype=np.uint32)

        if depth == 1:
            return vertices, triangles
        else:
            return recurse_sphere(vertices, triangles, depth-1)

    vertices, indices = recurse_sphere(v0, t0)
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    normals = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    uv_x = np.arctan2(vertices[:, 0], vertices[:, 2]) / np.pi * 0.5 + 0.5
    uv_y = np.arcsin(vertices[:, 1]) / np.pi + 0.5
    uvs = np.concatenate([uv_x[:, np.newaxis], uv_y[:, np.newaxis]], 1)

    return vertices, indices, normals, uvs


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
                 uvs=None, material=None
                 ):
        if colors is None:
            colors = np.ones_like(vertices)

        if normals is None:
            vertices, triangles, colors, uvs = self._make_triangles_unique(
                vertices, triangles, colors, uvs
            )
            normals = self._compute_normals(vertices, triangles)

        if material is None:
            material = Material(
                render_pass,
                'mesh',
                'mesh_vert.glsl',
                'mesh_frag.glsl',
                enable_texture=(uvs is not None),
            )
        self._material = material
        self._texture = None

        self._triangle_count = triangles.shape[0]

        self._material.set_uniform('indices', triangles.flatten())
        self._material.set_uniform('position', vertices)
        self._material.set_uniform('projection', projection)
        self._material.set_uniform('normal', normals)
        self._material.set_uniform('color', colors)

        if uvs is not None:
            self._material.shader.set_buffer('uv', uvs)

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

    def _make_triangles_unique(self, vertices, triangles, colors, uvs):
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

        if uvs is not None:
            uvs = make_component_unique(uvs)

        triangles = np.arange(vertices.shape[0]).reshape((-1, 3)).astype(np.uint32)

        return vertices, triangles, colors, uvs

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
            'wireframe_frag.glsl',
            enable_texture=False,
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
