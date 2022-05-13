from typing import Optional

from nanogui_sci3d import Texture, CubeMap
from scipy.interpolate import interp1d

from sci3d.colormaps import COOLWARM
from sci3d.window import Sci3DWindow
from sci3d.common import get_projection_matrix, Mesh
from sci3d.api.basicsurface import BasicSurface, BasicSurfaceApi, Params
import sci3d.common as common
import sci3d.materials as materials

import numpy as np

from sci3d.uithread import run_in_ui_thread
from sci3d.materials import Material


class SphericalHeatmapSurface(BasicSurface):
    def __init__(self,
                 window: Sci3DWindow,
                 common_params: Params,
                 points: np.ndarray,
                 scalars: Optional[np.ndarray],
                 resolution: int,
                 smoothing: float):
        self._resolution = resolution
        self._smoothing = smoothing
        self._colormap = interp1d(np.mgrid[0:1:1j*COOLWARM.shape[0]], COOLWARM, axis=0)

        # Convert from pixel coords to spherical
        # Then, from spherical coords to cartesian coords
        # TODO think about edge
        y_pix, x_pix = np.mgrid[0:self._resolution//2, 0:self._resolution]
        y_spherical = -((y_pix * 2 / self._resolution) * 2.0 - 1) * np.pi / 2.0
        y_cartesian = np.sin(y_spherical)
        radius = np.cos(y_spherical)

        x_spherical = ((x_pix / self._resolution) * 2.0 - 1) * np.pi
        x_cartesian = radius * np.sin(x_spherical)
        z_cartesian = radius * np.cos(x_spherical)

        self._cartesian = np.concatenate([
            x_cartesian[..., np.newaxis],
            y_cartesian[..., np.newaxis],
            z_cartesian[..., np.newaxis]
        ], 2)

        super(SphericalHeatmapSurface, self).__init__(window, common_params)

        vertices, triangles, normals, uvs = common.sphere(1.0)
        material = Material(
            window.render_pass,
            'mesh',
            'mesh_vert.glsl',
            'mesh_frag.glsl',
            enable_texture=True,
            enable_lighting=False
        )

        self._mesh = Mesh(
            window.render_pass,
            vertices, triangles, normals, np.ones_like(vertices),
            self._get_projection_matrix(),
            uvs=uvs, material=material
        )

        self._texture = Texture(
            Texture.PixelFormat.RGB,
            Texture.ComponentFormat.Float32,
            (self._resolution, self._resolution // 2),
            wrap_mode=Texture.WrapMode.ClampToEdge
        )
        self._mesh.get_material().shader.set_texture('tex_sampler', self._texture)
        self.post_init()

        self.set_points(points, scalars)

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

    def set_points(self,
                   points: np.ndarray,
                   scalars: Optional[np.ndarray]=None):
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        weights = np.ones((points.shape[0], 1), dtype=np.float32) \
            if scalars is None else scalars

        texture = np.zeros((self._resolution // 2, self._resolution))
        point_density = np.full_like(texture, 1e-6)
        for y_pix in range(self._resolution // 2):
            cartesian_y = self._cartesian[y_pix, ...]

            # Compute geodesic distance to given points
            geodist = np.sum(cartesian_y[:, np.newaxis, :] * points[np.newaxis, :, :], axis=-1)
            geodist = np.arccos(np.clip(geodist, -1, 1))
            geodist = np.exp(-0.5 * (geodist/self._smoothing)**2)

            texture[y_pix, :] = np.sum(geodist * weights.T, axis=1)
            point_density[y_pix, :] += np.sum(geodist, axis=1)

        if scalars is not None:
            # If scalars is provided, we want its average weighted by the density of points
            texture = texture / point_density

        texture = (
                (texture - np.min(texture)) / (np.max(texture) - np.min(texture))
        ).astype(np.float32).reshape([self._resolution//2, self._resolution])

        # Apply colormap
        texture = self._colormap(texture).astype(np.float32)
        self._texture.upload(texture)

    def _get_projection_matrix(self):
        return get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=self._window.camera_fov,
            hw_ratio=self._window.size()[1] / self._window.size()[0],
            scale_factor=self._window.scale_factor,
        )


class SphericalHeatmapApi(BasicSurfaceApi):
    _plot_drawer: SphericalHeatmapSurface

    def __init__(self, window: Sci3DWindow, plot_drawer: SphericalHeatmapSurface):
        super(SphericalHeatmapApi, self).__init__(window, plot_drawer)

    def set_points(self,
                   points: Optional[np.ndarray]):
        """
        Modify spherical heatmap

        The number of vertices does not need to be the same as the amount when the plot was first created.

        :param points: Rank 2 of shape [n_vertices, 3] and type float32
        """

        assert (points.ndim == 2)
        assert (points.shape[1] == 3)
        assert (points.dtype == np.float32)

        def impl():
            self._plot_drawer.set_points(points)

        run_in_ui_thread(impl)
