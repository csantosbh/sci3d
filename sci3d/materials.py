from nanogui_sci3d import RenderPass, Shader
from pkg_resources import resource_stream


class Material(object):
    def __init__(self,
                 render_pass: RenderPass,
                 name: str,
                 vertex_shader_file: str,
                 frag_shader_file: str,
                 enable_texture: bool,
                 enable_lighting: bool = True):
        def add_preprocessor_directives(shader):
            shader = shader.decode('utf-8').split('\n')
            if enable_texture:
                shader.insert(1, '#define ENABLE_TEXTURE')
            if enable_lighting:
                shader.insert(1, '#define ENABLE_LIGHTING')
            return '\n'.join(shader)

        vertex_shader = add_preprocessor_directives(
            resource_stream('sci3d', f'plottypes/shaders/{vertex_shader_file}').read()
        )
        fragment_shader = add_preprocessor_directives(
            resource_stream('sci3d', f'plottypes/shaders/{frag_shader_file}').read()
        )

        self._shader = Shader(
            render_pass,
            name,
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )

    def set_uniform(self, name, value):
        try:
            self.shader.set_buffer(name, value)
        except RuntimeError:
            pass

    @property
    def shader(self) -> Shader:
        return self._shader
