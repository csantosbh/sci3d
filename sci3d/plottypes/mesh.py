from typing import Optional
from sci3d.window import Sci3DWindow
from sci3d.common import get_projection_matrix, Mesh
from sci3d.api.basicsurface import BasicSurface, BasicSurfaceApi, Params
import sci3d.common as common
import sci3d.materials as materials

import numpy as np

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
    _plot_drawer: MeshSurface

    def __init__(self, window: Sci3DWindow, plot_drawer: MeshSurface):
        super(MeshApi, self).__init__(window, plot_drawer)

    def set_mesh(self,
                 vertices: Optional[np.ndarray] = None,
                 triangles: Optional[np.ndarray] = None,
                 normals: Optional[np.ndarray] = None,
                 colors: Optional[np.ndarray] = None):
        """
        Modify mesh components

        The number of vertices does not need to be the same as the amount when the plot was first created.

        :param vertices: Rank 2 of shape [n_vertices, 3] and type float32
        :param triangles: Rank 2 of shape [n_triangles, 3] and type uint32
        :param normals: Rank 2 of shape [n_vertices, 3] and type float32
        :param colors: Rank 2 of shape [n_vertices, 3] and type float32
        """

        assert (vertices.ndim == 2)
        assert (vertices.shape[1] == 3)
        assert (vertices.dtype == np.float32)

        assert (triangles.ndim == 2)
        assert (triangles.shape[1] == 3)
        assert (triangles.dtype == np.uint32)

        if normals is not None:
            assert (normals.ndim == 2)
            assert (normals.shape[0] == vertices.shape[0])
            assert (normals.shape[1] == 3)
            assert (normals.dtype == np.float32)

        if colors is not None:
            assert (colors.ndim == 2)
            assert (colors.shape[1] == 3)
            assert (colors.dtype == np.float32)

        def impl():
            self._plot_drawer.mesh_object.set_mesh(vertices, triangles, normals, colors)

        run_in_ui_thread(impl)
