#version 330
//#define CAMERA_ORTHOGONAL
in vec4 position_clip;
in vec4 position_world;

layout(location = 0) out vec4 color;
layout(location = 1) out vec4 out_surface_position;

uniform sampler3D scalar_field;
uniform mat4 object2camera;
uniform vec3 light_pos[4];
uniform vec3 light_color[4];


vec3 apply_lights(vec3 surface_pos, vec3 normal)
{
    vec3 output_color = vec3(0);

    for(int light_idx = 0; light_idx < 4; ++light_idx) {
        vec3 surface_to_light = light_pos[light_idx] - surface_pos;
        //vec3 surface_to_light = light_pos - surface_pos;
        float s2l_distance = length(surface_to_light);
        float lambertian_intensity = max(0, dot(
        normal, normalize(surface_to_light)
        )) / s2l_distance;
        output_color += lambertian_intensity * light_color[light_idx];
    }

    return output_color;
}

void main() {
    color = vec4(1, 1, 1, 1);

    color.rgb = position_clip.xyz;

    float depth_01 = position_clip.z / position_clip.w;

    gl_FragDepth = depth_01;
    out_surface_position = position_world;


    /*
    vec3 curr_pos_l = vec3(uv, 0.0);
    vec3 curr_pos_w = (object2camera * vec4(curr_pos_l, 1)).xyz;

    // Hide fragment if inside volume
    color.a = max(0, sign(sample_sdf(curr_pos_w.xyz)));

    // Setup camera
    #if defined(CAMERA_ORTHOGONAL)
    // orthogonal
    vec4 curr_dir = object2camera * vec4(0, 0, -1, 0);
    #else
    // perspective
    float focal_length = 1.0;
    vec4 curr_dir = object2camera * vec4(
    normalize(curr_pos_l - vec3(0.5, 0.5, focal_length)),
    0
    );
    #endif

    // Get surface geometry
    vec4 iso_intersection = intersect_isocurve_0(curr_pos_w, curr_dir.xyz);
    vec3 surface_pos = iso_intersection.xyz;
    float sdf_value = iso_intersection.w;
    float eps = 0.01;
    // Hide pixels that do not contain the isosurface
    color.a *= smoothstep(2*eps, eps, iso_intersection.w);
    vec3 normal = normalize(gradient(surface_pos));

    // Apply lighting
    color.rgb = apply_lights(surface_pos, normal);

    out_surface_position = vec4(surface_pos, 1);
    */
}
