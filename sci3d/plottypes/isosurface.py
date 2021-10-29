from pathlib import Path
from sci3d.example2 import Sci3DWindow

import numpy as np
from nanogui import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D, \
    Matrix4f, glfw, icons


class Isosurface(object):
    def __init__(self, window: Sci3DWindow, isosurface):
        curr_path = Path(__file__).parent.resolve()

        with open(curr_path / 'shaders/isosurface_vert.glsl') as f:
            vertex_shader = f.read()

        with open(curr_path / 'shaders/isosurface_frag.glsl') as f:
            fragment_shader = f.read()

        self._shader = Shader(
            # TODO create public accessor
            window._render_pass,
            "isosurface",
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )

        self._texture = Texture3D(
            Texture.PixelFormat.R,
            Texture.ComponentFormat.Float32,
            isosurface.shape,
            wrap_mode=Texture.WrapMode.ClampToEdge
        )

        light_pos = np.eye(4, 3).astype(np.float32)
        light_color = np.eye(4, 3).astype(np.float32)
        self._shader.set_buffer("scale_factor", np.array(1.0, dtype=np.float32))
        self._shader.set_buffer("light_pos[0]", light_pos.flatten())
        self._shader.set_buffer("light_color[0]", light_color.flatten())

        self._shader.set_buffer("indices", np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32))
        self._shader.set_buffer("position", np.array(
            [[-1, -1, 0],
             [1, -1, 0],
             [1, 1, 0],
             [-1, 1, 0]],
            dtype=np.float32
        ))

        self._window = window
        self._shader.set_texture3d("scalar_field", self._texture)

        self._texture.upload(isosurface)
        self._shader.set_buffer(
            "image_resolution", np.array(isosurface.shape[0], dtype=np.float32))

    def draw(self):
        s = self._window.size()
        view_scale = Matrix4f.scale([1, s[0] / s[1], 1])
        mvp = view_scale
        self._shader.set_buffer("mvp", np.float32(mvp).T)
        # TODO make public accessor fow _camera_matrix
        self._shader.set_buffer("object2camera", self._window._camera_matrix.T)
        # TODO make public accessor fow _camera_matrix
        self._shader.set_buffer(
            "scale_factor", np.array(0.95**self._window._scale_power, dtype=np.float32))

        with self._shader:
            self._shader.draw_array(Shader.PrimitiveType.Triangle, 0, 6, True)

