from typing import Optional, Tuple
from pathlib import Path

import numpy as np

from nanogui import Shader


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
    def __init__(self, render_pass, vertices, triangles, normals, colors, projection):
        self._num_lights = 4

        curr_path = Path(__file__).parent.resolve()

        if colors is None:
            colors = np.ones_like(vertices)

        if normals is None:
            vertices, triangles, colors = self._make_triangles_unique(
                vertices, triangles, colors
            )
            normals = self._compute_normals(vertices, triangles)

        with open(curr_path / 'plottypes/shaders/mesh_vert.glsl') as f:
            vertex_shader = f.read()

        with open(curr_path / 'plottypes/shaders/mesh_frag.glsl') as f:
            fragment_shader = f.read()

        self._shader = Shader(
            render_pass,
            "mesh",
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )
        self._texture = None

        self._triangle_count = triangles.shape[0]
        self._shader.set_buffer('indices', triangles.flatten())
        self._shader.set_buffer('position', vertices)
        self._shader.set_buffer('projection', projection)
        self._shader.set_buffer('normal', normals)
        self._shader.set_buffer('color', colors)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if light_pos.shape[0] != self._num_lights:
            light_pos = np.pad(light_pos, [[0, self._num_lights - light_pos.shape[0]], [0, 0]])

        if light_color.shape[0] != self._num_lights:
            light_color = np.pad(light_color, [[0, self._num_lights - light_color.shape[0]], [0, 0]])

        self._shader.set_buffer("light_pos[0]", light_pos.flatten())
        self._shader.set_buffer("light_color[0]", light_color.flatten())

    def set_mesh(self,
                 triangles: Optional[np.ndarray],
                 vertices: Optional[np.ndarray],
                 normals: Optional[np.ndarray]):
        if triangles is not None:
            self._triangle_count = triangles.shape[0]
            self._shader.set_buffer("indices", triangles.flatten())

        if vertices is not None:
            self._shader.set_buffer("position", vertices)

        if normals is not None:
            self._shader.set_buffer("normal", normals)

    def draw(self,
             object2world: np.ndarray,
             world2camera: np.ndarray,
             projection: np.ndarray):
        self._shader.set_buffer("object2world", object2world.T)
        self._shader.set_buffer("world2camera", world2camera.T)
        self._shader.set_buffer("projection", projection.T)

        with self._shader:
            self._shader.draw_array(
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


