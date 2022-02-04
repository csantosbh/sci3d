#version 330
in vec4 position_clip;
in vec4 position_world;
in vec3 color_vert;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 out_surface_position;

uniform sampler3D scalar_field;
uniform mat4 object2camera;
uniform vec3 light_pos[4];


void main() {
    color = vec4(color_vert, 1);
    out_surface_position = vec4(0);
}
