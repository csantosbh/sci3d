#version 330
in vec4 position_clip;
in vec4 position_world;
in vec3 normal_world;
in vec3 color_vert;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 out_surface_position;

uniform sampler3D scalar_field;
uniform mat4 object2camera;
uniform vec3 light_pos[4];
uniform vec3 light_color[4];


void main() {
    color = vec4(0, 0, 0, 1);

    vec3 normal_world_n = normalize(normal_world);
    const int num_lights = 4;
    for(int light_idx = 0; light_idx < num_lights; ++light_idx) {
        // Lambertian component
        vec3 surface_to_light = light_pos[light_idx] - position_world.xyz;
        float s2l_distance = length(surface_to_light);
        float lambertian_intensity = max(0, dot(
            normal_world_n, normalize(surface_to_light)
        ));
        color.rgb += lambertian_intensity * light_color[light_idx] * color_vert;
    }

    out_surface_position = position_world;
}
