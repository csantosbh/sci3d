#version 330
//#define CAMERA_ORTHOGONAL
in vec2 uv;
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 out_surface_position;

uniform sampler3D scalar_field;
uniform mat4 object2camera;
uniform float image_resolution;
uniform vec3 light_pos[4];
uniform vec3 light_color[4];

float sample_sdf(vec3 coordinate) {
    // Convert coordinate to left handed notation (texture coord system)
    vec3 left_handed_coord = vec3(coordinate.x, coordinate.y, -coordinate.z);
    return texture(scalar_field, left_handed_coord).r;
}

vec3 get_start_near_volume(vec3 start, vec3 ray_direction) {
    // Make rays start at the edge of the volume, even if camera is far away
    vec3 far_corner = vec3(1, 1, -1);
    float alpha = max(0,
        min((-start.x) / ray_direction.x,
            (far_corner.x - start.x) / ray_direction.x)
    );
    start += alpha * ray_direction;

    alpha = max(0,
        min((-start.y) / ray_direction.y,
            (far_corner.y - start.y) / ray_direction.y)
    );
    start += alpha * ray_direction;

    alpha = max(0,
        min((-start.z) / ray_direction.z,
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
}
