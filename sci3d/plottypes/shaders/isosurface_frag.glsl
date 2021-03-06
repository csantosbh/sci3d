#version 330
in vec2 screen_coords_01;
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 out_surface_position;

uniform sampler3D scalar_field;
uniform mat4 world2object;
uniform mat4 camera2object;
uniform float image_resolution;
uniform float camera_fov;
uniform vec3 light_pos[4];
uniform vec3 light_color[4];

float sample_sdf(vec3 coordinate) {
    // Convert coordinate to left handed notation (texture coord system)
    vec3 left_handed_coord = vec3(coordinate.x, coordinate.y, -coordinate.z) + 0.5;
    return texture(scalar_field, left_handed_coord).r;
}

vec3 get_start_near_volume(vec3 start, vec3 ray_direction) {
    // Make rays start at the edge of the volume, even if camera is far away
    vec3 far_corner = vec3(0.5, 0.5, 0.5);
    vec3 near_corner = vec3(-0.5, -0.5, -0.5);

    float alpha = max(0,
        min((near_corner.x - start.x) / ray_direction.x,
            (far_corner.x - start.x) / ray_direction.x)
    );
    start += alpha * ray_direction;

    alpha = max(0,
        min((near_corner.y - start.y) / ray_direction.y,
            (far_corner.y - start.y) / ray_direction.y)
    );
    start += alpha * ray_direction;

    alpha = max(0,
        min((near_corner.z - start.z) / ray_direction.z,
            (far_corner.z - start.z) / ray_direction.z)
    );
    start += alpha * ray_direction;

    return start;
}

vec4 intersect_isocurve_0(vec3 start, vec3 ray_direction) {
    float max_ray_coverage = sqrt(3.0);
    float step_count = image_resolution;
    vec3 ray_pos = get_start_near_volume(start, ray_direction);
    float alpha = 0;
    float step_size = max_ray_coverage / step_count;

    float sdf;
    float was_ray_previously_outside = 1;

    for(int i = 0; i < step_count; ++i) {
        sdf = sample_sdf(ray_pos + ray_direction * alpha);
        float is_ray_outside = float(sdf >= 0);
        float step_direction = mix(-1, 1, is_ray_outside);
        float ray_crossed_isocurve = float(
            is_ray_outside != was_ray_previously_outside
        );
        step_size = mix(step_size, step_size * 0.5, ray_crossed_isocurve);
        was_ray_previously_outside = is_ray_outside;
        alpha += step_direction * step_size;

        if (step_size < 1e-6) break;
    }
    return vec4(ray_pos + ray_direction * alpha, sdf);
}

float derivative(vec3 pos, vec3 eps) {
    return sample_sdf(pos + eps) - sample_sdf(pos - eps);
}

vec3 gradient(vec3 pos) {
    float eps = 1.0 / image_resolution;

    float dx = derivative(pos, vec3(eps, 0, 0));
    float dy = derivative(pos, vec3(0, eps, 0));
    float dz = derivative(pos, vec3(0, 0, eps));

    return vec3(dx, dy, dz);
}

vec3 apply_lights(vec3 surface_pos, vec3 normal)
{
    vec3 output_color = vec3(0);

    const int num_lights = 4;
    for(int light_idx = 0; light_idx < num_lights; ++light_idx) {
        vec3 surface_to_light = light_pos[light_idx] - surface_pos;
        float s2l_distance = length(surface_to_light);
        float lambertian_intensity = max(0, dot(
            normal, normalize(surface_to_light)
        )) / s2l_distance;
        output_color += lambertian_intensity * light_color[light_idx];
    }

    return output_color;
}

void main() {
    float near_plane_side = tan(camera_fov * 0.5);
    vec3 curr_pos_l = vec3(screen_coords_01 * near_plane_side, -1);
    vec3 curr_pos_w = (camera2object * vec4(0, 0, 0, 1)).xyz;
    vec3 cam_forward = (camera2object * vec4(0, 0, -1, 0)).xyz;

    // Hide fragment if inside volume
    color.a = max(0, sign(sample_sdf(curr_pos_w.xyz)));

    // perspective direction
    vec4 curr_dir = camera2object * vec4(
        normalize(curr_pos_l),
        0
    );

    // Get surface geometry
    vec4 iso_intersection = intersect_isocurve_0(curr_pos_w, curr_dir.xyz);
    vec3 surface_pos = iso_intersection.xyz;
    float sdf_value = iso_intersection.w;
    float eps = 0.01;
    // Hide pixels that do not contain the isosurface
    color.a *= smoothstep(2 * eps, eps, iso_intersection.w);
    vec3 normal = normalize(gradient(surface_pos));

    // Apply lighting
    color.rgb = apply_lights(surface_pos, normal);

    out_surface_position = vec4(surface_pos, 1);

    // Map depth to NDC coordinates
    const float near = 0.1;
    const float far = 1e3;
    // TODO make this uniform
    vec2 depth_remap = vec2(-(far + near) / (far - near), -2*far*near / (far - near));

    float fragment_depth = -dot(cam_forward, surface_pos - curr_pos_w);
    // Right now, z is in the range [-1, 1]
    float depth_minus1_1 = (fragment_depth * depth_remap.x + depth_remap.y) / -fragment_depth;
    // Now we map it to [0, 1], which is the range of the depth buffer
    gl_FragDepth = depth_minus1_1 * 0.5 + 0.5;
}
