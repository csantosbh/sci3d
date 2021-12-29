from typing import Optional
from pathlib import Path
from sci3d.window import Sci3DWindow
from sci3d.common import get_projection_matrix, Mesh
from sci3d.api.basicsurface import BasicSurface
from sci3d.common import BoundingBox

import numpy as np
from nanogui import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D, \
    Matrix4f

from sci3d.uithread import run_in_ui_thread


class MeshSurface(BasicSurface):
    def __init__(self,
                 window: Sci3DWindow,
                 vertices: np.ndarray,
                 triangles: np.ndarray,
                 normals: Optional[np.ndarray],
                 colors: Optional[np.ndarray],
                 pose: Optional[np.ndarray]):
        super(MeshSurface, self).__init__(window)

        self._mesh = Mesh(
            # TODO create public accessor
            window._render_pass,
            vertices, triangles, normals, colors, self._get_projection_matrix()
        )

        # TODO make lights part of the Window instead
        light_pos = np.eye(4, 3).astype(np.float32)
        light_color = np.eye(4, 3).astype(np.float32)
        self._mesh.set_lights(light_pos, light_color)
        self._object2world = pose if pose is not None else np.eye(4, dtype=np.float32)

    @property
    def mesh_object(self) -> Mesh:
        return self._mesh

    def get_bounding_box(self) -> BoundingBox:
        return self._mesh.get_bounding_box()

    def draw(self):
        self._mesh.draw(
            self._object2world,
            self._window.world2camera(),
            self._get_projection_matrix()
        )

    def _get_projection_matrix(self):
        return get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=self._window.camera_fov,
            hw_ratio=self._window.size()[1] / self._window.size()[0],
            scale_factor=self._window.scale_factor,
        )


class MeshApi(object):
    def __init__(self, window: Sci3DWindow, plot_drawer: MeshSurface):
        self._window = window
        self._plot_drawer = plot_drawer

    def set_title(self, title):
        self._window.set_caption(title)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if not self._window.visible():
            return

        def impl():
            self._plot_drawer.mesh_object.set_lights(light_pos, light_color)

        run_in_ui_thread(impl)

    def set_mesh(self,
                 triangles: Optional[np.ndarray] = None,
                 vertices: Optional[np.ndarray] = None,
                 normals: Optional[np.ndarray] = None):
        def impl():
            self._plot_drawer.mesh_object.set_mesh(triangles, vertices, normals)

        run_in_ui_thread(impl)
