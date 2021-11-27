from typing import Optional
from pathlib import Path
from sci3d.window import Sci3DWindow
from sci3d.common import get_projection_matrix

import numpy as np
from nanogui import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D, \
    Matrix4f

from sci3d.uithread import run_in_ui_thread


class Mesh(object):
    def __init__(self, window: Sci3DWindow, parameters: dict):
        vertices: np.ndarray = parameters['vertices']
        triangles: np.ndarray = parameters['triangles']

        self._num_lights = 4

        curr_path = Path(__file__).parent.resolve()

        with open(curr_path / 'shaders/mesh_vert.glsl') as f:
            vertex_shader = f.read()

        with open(curr_path / 'shaders/mesh_frag.glsl') as f:
            fragment_shader = f.read()

        self._shader = Shader(
            # TODO create public accessor
            window._render_pass,
            "mesh",
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )
        self._texture = None
        self._window = window

        light_pos = np.eye(4, 3).astype(np.float32)
        light_color = np.eye(4, 3).astype(np.float32)
        #self.set_lights(light_pos, light_color)

        self._triangle_count = triangles.shape[0]
        self._shader.set_buffer("indices", triangles.flatten())
        self._shader.set_buffer("position", vertices)

        self._shader.set_buffer("projection", self._get_projection_matrix().T)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if light_pos.shape[0] != self._num_lights:
            light_pos = np.pad(light_pos, [[0, self._num_lights - light_pos.shape[0]], [0, 0]])
        if light_color.shape[0] != self._num_lights:
            light_color = np.pad(light_color, [[0, self._num_lights - light_color.shape[0]], [0, 0]])

        self._shader.set_buffer("light_pos[0]", light_pos.flatten())
        self._shader.set_buffer("light_color[0]", light_color.flatten())

    def set_mesh(self,
                 triangles: Optional[np.ndarray],
                 vertices: Optional[np.ndarray]):
        if triangles is not None:
            self._triangle_count = triangles.shape[0]
            self._shader.set_buffer("indices", triangles.flatten())

        if vertices is not None:
            self._shader.set_buffer("position", vertices)

    def draw(self):
        self._shader.set_buffer("object2camera", self._window.world2camera().T)
        self._shader.set_buffer("projection", self._get_projection_matrix().T)

        with self._shader:
            self._shader.draw_array(
                Shader.PrimitiveType.Triangle, 0, self._triangle_count * 3, True)

    def _get_projection_matrix(self):
        return get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=self._window.camera_fov,
            hw_ratio=self._window.size()[1] / self._window.size()[0],
            scale_factor=self._window.scale_factor,
        )


class MeshApi(object):
    def __init__(self, window: Sci3DWindow, plot_drawer: Mesh):
        self._window = window
        self._plot_drawer = plot_drawer

    def set_title(self, title):
        self._window.set_caption(title)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if not self._window.visible():
            return

        def impl():
            self._plot_drawer.set_lights(light_pos, light_color)

        run_in_ui_thread(impl)

    def set_mesh(self,
                 triangles: Optional[np.ndarray] = None,
                 vertices: Optional[np.ndarray] = None):
        def impl():
            self._plot_drawer.set_mesh(triangles, vertices)

        run_in_ui_thread(impl)
