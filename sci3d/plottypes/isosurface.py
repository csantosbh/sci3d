from pathlib import Path

import numpy as np
from nanogui_sci3d import Shader, Texture, Texture3D, Matrix4f

from sci3d.window import Sci3DWindow
from sci3d.uithread import run_in_ui_thread
from sci3d.api.basicsurface import BasicSurface, BasicSurfaceApi, Params
import sci3d.common as common
from sci3d.materials import Material


class Isosurface(BasicSurface):
    def __init__(self,
                 window: Sci3DWindow,
                 common_params: Params,
                 volume: np.ndarray):
        super(Isosurface, self).__init__(window, common_params)

        self._material = Material(
            window.render_pass,
            'isosurface',
            'isosurface_vert.glsl',
            'isosurface_frag.glsl',
        )
        self._texture = None

        self._material.shader.set_buffer("indices", np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32))
        self._material.shader.set_buffer("position", np.array(
            [[-1, -1, 0],
             [1, -1, 0],
             [1, 1, 0],
             [-1, 1, 0]],
            dtype=np.float32
        ))

        self.set_isosurface(volume)
        self._bounding_box = common.BoundingBox(
            np.array([-0.5, -0.5, -0.5], dtype=np.float32),
            np.array([0.5, 0.5, 0.5], dtype=np.float32),
        )

        self._material.shader.set_buffer("scale_factor", np.array(1.0, dtype=np.float32))

        self.post_init()

    def get_bounding_box(self) -> common.BoundingBox:
        return self._bounding_box

    def get_material(self) -> Material:
        return self._material

    def set_isosurface(self, volume):
        self._window.make_context_current()
        if self._texture is None or self._texture.size() != volume.shape:
            self._texture = Texture3D(
                Texture.PixelFormat.R,
                Texture.ComponentFormat.Float32,
                volume.shape,
                wrap_mode=Texture.WrapMode.ClampToEdge
            )

            self._material.shader.set_texture3d("scalar_field", self._texture)
            self._material.shader.set_buffer(
                "image_resolution", np.array(volume.shape[0], dtype=np.float32))

        self._texture.upload(volume)

    def draw(self):
        s = self._window.size()
        view_scale = Matrix4f.scale([1, s[0] / s[1], 1])
        mvp = view_scale

        self._material.shader.set_buffer("mvp", np.float32(mvp).T)
        world2object = common.inverse_affine(self._object_rotation, self._object_position)
        camera2object = world2object @ self._window.camera2world()
        self._material.shader.set_buffer("camera2object", camera2object.T)
        self._material.shader.set_buffer("camera_fov", self._window.camera_fov)
        self._material.shader.set_buffer("scale_factor", self._window.scale_factor)

        with self._material.shader:
            self._material.shader.draw_array(Shader.PrimitiveType.Triangle, 0, 6, True)


class IsosurfaceApi(BasicSurfaceApi):
    _plot_drawer: Isosurface

    def __init__(self, window: Sci3DWindow, plot_drawer: Isosurface):
        super(IsosurfaceApi, self).__init__(window, plot_drawer)

    def set_isosurface(self, volume: np.ndarray):
        """
        Update volumetric scalar field of 0-level set plot

        :param volume: Input volume. Must be a rank 3 tensor of shape [nz, ny, nx] and float32 type TODO confirm zyx order
        """
        assert(volume.ndim == 3)
        assert(volume.dtype == np.float32)
        # TODO: Currently we only support cube volumes
        assert(volume.shape[0] == volume.shape[1] == volume.shape[2])

        if not self._window.visible():
            return

        def impl():
            self._plot_drawer.set_isosurface(volume)

        run_in_ui_thread(impl)
