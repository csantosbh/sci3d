from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

from nanogui_sci3d import RenderPass, Shader


class Material(object):
    def __init__(self,
                 render_pass: RenderPass,
                 name: str,
                 vertex_shader_file: str,
                 frag_shader_file: str):
        curr_path = Path(__file__).parent.resolve() / 'plottypes/shaders'

        with open(curr_path / vertex_shader_file) as f:
            vertex_shader = f.read()

        with open(curr_path / frag_shader_file) as f:
            fragment_shader = f.read()

        self._shader = Shader(
            render_pass,
            name,
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )

    @property
    def shader(self) -> Shader:
        return self._shader
