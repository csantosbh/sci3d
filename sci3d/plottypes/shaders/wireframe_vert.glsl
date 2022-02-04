#version 330
uniform mat4 object2world;
uniform mat4 world2camera;
uniform mat4 projection;

in vec3 position;
in vec3 color;

out vec4 position_world;
out vec4 position_clip;
out vec4 gl_Position;
out vec3 color_vert;

void main() {
    position_world = object2world * vec4(position, 1.0);
    vec4 position_cam = world2camera * position_world;
    position_clip = projection * position_cam;
    gl_Position = position_clip;

    color_vert = color;
}
