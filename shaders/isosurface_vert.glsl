#version 330
uniform mat4 mvp;
uniform float scale_factor;

in vec3 position;
out vec2 uv;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    uv = scale_factor * (position.xy * 0.5) + 0.5;
}
