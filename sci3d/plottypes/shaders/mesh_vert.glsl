#version 330
uniform mat4 object2camera;
uniform mat4 projection;
uniform float scale_factor;

in vec3 position;

out vec4 position_clip;
out vec4 position_world;

void main() {
    position_world = vec4(position, 1.0);
    vec4 position_cam = object2camera * position_world;
    position_clip = projection * position_cam;
    gl_Position = position_clip;
}
