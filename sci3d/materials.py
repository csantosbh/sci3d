from nanogui_sci3d import RenderPass, Shader
from pkg_resources import resource_stream


class Material(object):
    def __init__(self,
                 render_pass: RenderPass,
                 name: str,
                 vertex_shader_file: str,
                 frag_shader_file: str):
        vertex_shader = resource_stream('sci3d', f'plottypes/shaders/{vertex_shader_file}').read()
        fragment_shader = resource_stream('sci3d', f'plottypes/shaders/{frag_shader_file}').read()

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
