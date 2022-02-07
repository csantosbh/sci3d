from typing import Optional
from pathlib import Path
from sci3d.window import Sci3DWindow
from sci3d.common import get_projection_matrix, Mesh
from sci3d.api.basicsurface import BasicSurface, BasicSurfaceApi, Params
import sci3d.common as common
import sci3d.materials as materials

import numpy as np
from nanogui import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D, \
    Matrix4f

from sci3d.uithread import run_in_ui_thread


class MeshSurface(BasicSurface):
    def __init__(self,
                 window: Sci3DWindow,
                 common_params: Params,
                 vertices: np.ndarray,
                 triangles: np.ndarray,
                 normals: Optional[np.ndarray],
                 colors: Optional[np.ndarray]):
        super(MeshSurface, self).__init__(window, common_params)

        self._mesh = Mesh(
            window.render_pass,
            vertices, triangles, normals, colors, self._get_projection_matrix()
        )

        self.post_init()

    def get_material(self) -> materials.Material:
        return self._mesh.get_material()

    @property
    def mesh_object(self) -> Mesh:
        return self._mesh

    def get_bounding_box(self) -> common.BoundingBox:
        return self._mesh.get_bounding_box()

    def draw(self):
        object2world = common.forward_affine(self._object_rotation, self._object_position)
        # TODO move to Mesh
        self._mesh.get_material().shader.set_buffer("object2world", object2world.T)

        self._mesh.draw(
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


class MeshApi(BasicSurfaceApi):
    def __init__(self, window: Sci3DWindow, plot_drawer: MeshSurface):
        super(MeshApi, self).__init__(window, plot_drawer)

    def set_mesh(self,
                 triangles: Optional[np.ndarray] = None,
                 vertices: Optional[np.ndarray] = None,
                 normals: Optional[np.ndarray] = None):
        def impl():
            self._plot_drawer.mesh_object.set_mesh(triangles, vertices, normals)

        run_in_ui_thread(impl)
