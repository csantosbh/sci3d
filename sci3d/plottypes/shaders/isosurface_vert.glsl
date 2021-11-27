#version 330
uniform mat4 mvp;
uniform float scale_factor;

in vec3 position;
out vec2 screen_coords_01;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    screen_coords_01 = scale_factor * position.xy;
}
