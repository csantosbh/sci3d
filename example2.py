#!/usr/bin/env python

# python/example1.py -- Python version of an example application that shows
# how to use the various widget classes. For a C++ implementation, see
# '../src/example1.cpp'.
#
# NanoGUI was developed by Wenzel Jakob <wenzel@inf.ethz.ch>.
# The widget drawing code is based on the NanoVG demo application
# by Mikko Mononen.
#
# All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE.txt file.
import queue
import threading

import nanogui
import math
import time
import gc
import numpy as np
import icecream

icecream.install()
ic.configureOutput(includeContext=True)

import nanogui.nanovg as nanovg
from nanogui import Color, ColorPicker, Screen, Window, GroupLayout, \
                    BoxLayout, ToolButton, Label, Button, Widget, \
                    Popup, PopupButton, CheckBox, MessageDialog, \
                    VScrollPanel, ImagePanel, ImageView, ComboBox, \
                    ProgressBar, Slider, TextBox, ColorWheel, Graph, \
                    GridLayout, Alignment, Orientation, TabWidget, \
                    IntBox, RenderPass, Shader, Texture, Texture3D, Matrix4f, \
                    Vector2i, Canvas, Theme

import numpy
from nanogui import glfw, icons

# A simple counter, used for dynamic tab creation with TabWidget callback
counter = 1


class Tooltip(object):
    def __init__(self,
                 tip_position_screen,
                 tip_position_world,
                 box_tip_distance=10,
                 tip_radius=4,
                 tip_color=Color(48, 48, 48, 192),
                 text_color=Color(255, 255, 255, 255),
                 nvg_context=None):
        super(Tooltip, self).__init__()
        self.tip_position_screen = tip_position_screen
        self.tip_position_world = tip_position_world
        self._margin = (5, 5)
        self._tip_color = tip_color
        self._text_color = text_color
        self._tip_radius = tip_radius
        self._is_moving_tip = None
        self._is_moving_window = None
        self._nvg_context = nvg_context
        self._size = np.zeros((2,))

        # Close icon
        self._close_icon_size = 10
        self._close_icon = eval(f'u"\\u{icons.FA_TIMES_CIRCLE:04x}"')
        # [minx, miny, maxx, maxy]
        self._close_icon_bounds = None
        self.destroy_requested_cbks = []

        theme = Theme(self._nvg_context)
        theme.m_window_drop_shadow_size = 3
        self.caption = None
        # [minx, miny, maxx, maxy]
        self.caption_bounds = None
        self.set_caption(self._build_caption())

        self._position = tip_position_screen - Vector2i(self._size[0] // 2,
                                                        box_tip_distance+int(self._size[1]))

    def set_caption(self, caption):
        # Get caption bounds
        self.caption = caption
        self.caption_bounds = self._nvg_context.TextBoxBounds(
            0.0, 0.0, 1e3, self.caption
        )

        # Get close icon bounds
        self._nvg_context.Save()
        self._nvg_context.FontFace("icons")
        self._nvg_context.FontSize(self._close_icon_size)
        close_icon_bounds = self._nvg_context.TextBounds(0, 0, self._close_icon)
        self._nvg_context.Restore()

        self._close_icon_bounds = [
            self.caption_bounds[2] - close_icon_bounds[0] + 2 * self._margin[0],
            -self.caption_bounds[1] + self._margin[1],
            0, 0
        ]
        close_icon_size = [
            int(close_icon_bounds[2] - close_icon_bounds[0]),
            int(close_icon_bounds[3] - close_icon_bounds[1])
        ]
        self._close_icon_bounds[2] = self._close_icon_bounds[0] + close_icon_size[0]
        self._close_icon_bounds[3] = self._close_icon_bounds[1] + close_icon_size[1]

        # Caption window size
        self._size = np.array([
            int(self.caption_bounds[2] - self.caption_bounds[0]) + 3*self._margin[0] + close_icon_size[0],
            int(self.caption_bounds[3] - self.caption_bounds[1]) + 2*self._margin[1]
        ])
        pass

    def _is_mouse_over_tip(self, position):
        return np.linalg.norm(position - self.tip_position_screen) <= self._tip_radius

    def _is_mouse_over_window(self, position):
        width, height = self._size

        return self._position[0] <= position[0] <= (self._position[0] + width) and \
               self._position[1] <= position[1] <= (self._position[1] + height)

    def _is_mouse_over_close_icon(self, position):
        bound_x = [
            self._position[0] + self._close_icon_bounds[0],
            self._position[0] + self._close_icon_bounds[2],
        ]
        bound_y = [
            self._position[1] + 2*self._close_icon_bounds[1] - self._close_icon_bounds[3],
            self._position[1] + self._close_icon_bounds[1],
        ]

        return bound_x[0] <= position[0] <= bound_x[1] and \
               bound_y[0] <= position[1] <= bound_y[1]

    def contains(self, position):
        #window_contains = super(Tooltip, self).contains(position)
        return self._is_mouse_over_window(position) or self._is_mouse_over_tip(position)

    def mouse_button_event(self, p, button, down, modifiers):
        #super(Tooltip, self).mouse_button_event(p, button, down, modifiers)

        if self._is_mouse_over_close_icon(p):
            [cbk(self) for cbk in self.destroy_requested_cbks]
            return True
        elif down:
            self._is_moving_tip = self._is_mouse_over_tip(p)
            self._is_moving_window = self._is_mouse_over_window(p)
        else:
            self._is_moving_tip = self._is_moving_window = False

        return self._is_moving_tip or self._is_moving_window

    def _build_caption(self):
        return (f'({self.tip_position_world[0]:.4f},'
                f' {self.tip_position_world[1]:.4f},'
                f' {self.tip_position_world[2]:.4f})')

    def set_tip_position_world(self, tip_position_world):
        self.tip_position_world = tip_position_world
        self.set_caption(self._build_caption())

    def mouse_drag_event(self, p, rel, button, modifiers, rt_pos_data):
        if self._is_moving_tip:
            self.tip_position_screen += rel
            # Update world position and label
            tip_position_world = np.concatenate([rt_pos_data[
                self.tip_position_screen[1],
                self.tip_position_screen[0],
                :
            ], [1]])
            self.set_tip_position_world(tip_position_world)
        elif self._is_moving_window:
            self._position = self._position + rel

        return self._is_moving_window or self._is_moving_tip

    def draw(self):
        nvg = self._nvg_context

        # Temporarily disable scissoring
        nvg.Save()
        nvg.Reset()

        # Draw data point
        nvg.BeginPath()
        nvg.Circle(
            self.tip_position_screen[0], self.tip_position_screen[1], self._tip_radius
        )
        box_center = self._position + self._size / 2
        tip_box_dir = np.array(box_center - self.tip_position_screen)
        line_tip_intersection = (
                tip_box_dir / np.linalg.norm(tip_box_dir) * self._tip_radius
        ).astype(np.int32)
        nvg.MoveTo(*box_center)
        nvg.LineTo(*(self.tip_position_screen + line_tip_intersection))
        nvg.StrokeColor(self._tip_color)
        nvg.StrokeWidth(2)

        # Restore scissoring
        nvg.Stroke()
        nvg.Restore()

        #super(Tooltip, self).draw(nvg)
        nvg.BeginPath()
        nvg.Rect(*self._position, *self._size)
        nvg.FillColor(self._tip_color)
        nvg.Fill()

        # Draw text
        nvg.BeginPath()
        nvg.FillColor(self._text_color)
        curr_pos = self._position
        nvg.TextBox(
            curr_pos[0] - self.caption_bounds[0] + self._margin[0],
            curr_pos[1] - self.caption_bounds[1] + self._margin[1],
            1e3,
            self.caption
        )

        # Draw close button
        nvg.Save()
        nvg.FontFace("icons")
        nvg.FontSize(self._close_icon_size)
        nvg.TextBox(
            curr_pos[0] + self._close_icon_bounds[0],
            curr_pos[1] + self._close_icon_bounds[1],
            1e3,
            self._close_icon
        )
        nvg.Restore()

        nvg.Stroke()


class TestApp(Screen):
    def setup_toolbar(self):
        self._toolbar = Window(self, "")
        self._toolbar.set_position((0, 0))

        toolbar = Widget(self._toolbar)
        padding = 2
        layout = BoxLayout(Orientation.Horizontal, Alignment.Middle, padding, 6)
        toolbar.set_layout(layout)

        def create_tooltip(p, button, down, modifiers):
            if down:
                make_tooltip_btn.set_pushed(False)
                tip_position_world = np.concatenate([
                    self.rt_pos_data[p[1], p[0], :],
                    [1]
                ])
                tooltip = Tooltip(
                    tip_position_screen=p, tip_position_world=tip_position_world,
                    nvg_context=self.nvg_context()
                )
                tooltip.destroy_requested_cbks.append(lambda tt: self._tooltips.remove(tt))
                self._tooltips.append(tooltip)
                self._handle_mouse_down = None

        def cbk_handler():
            self._handle_mouse_down = create_tooltip

        make_tooltip_btn = ToolButton(toolbar, icons.FA_CROSSHAIRS)
        make_tooltip_btn.set_callback(cbk_handler)

        self.perform_layout()
        btn_size = make_tooltip_btn.size()
        self._toolbar.set_size((2*padding + self.size()[0], 2*padding + btn_size[1]))

    def __init__(self):
        super(TestApp, self).__init__((1024, 768), "NanoGUI Test")
        self.shader = None
        self._handle_mouse_down = None

        self._toolbar = None
        self._tooltips = []
        self._scale_power = 0

        self.setup_toolbar()
        self.perform_layout()

        self.rt_color = Texture(
            # TODO we dont need rgba
            Texture.PixelFormat.RGBA,
            Texture.ComponentFormat.UInt8,
            # TODO recreate texture when size changes
            self.size(),
            # TODO using shaderread has implications that I'm not sure I want
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self.rt_position = Texture(
            Texture.PixelFormat.RGB,
            Texture.ComponentFormat.Float32,
            # TODO recreate texture when size changes
            self.size(),
            # TODO using shaderread has implications that I'm not sure I want
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self.render_pass = RenderPass([self.rt_color, self.rt_position], blit_target=self)
        self.render_pass.set_clear_color(0, Color(0.3, 0.3, 0.32, 1.0))

        self.rt_pos_data = None

        # We currently only support opengl
        assert(nanogui.api == 'opengl')

        vertex_shader = """
        #version 330
        uniform mat4 mvp;
        uniform float scale_factor;
        
        in vec3 position;
        out vec2 uv;
        
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            uv = scale_factor * (position.xy * 0.5) + 0.5;
        }"""

        fragment_shader = """
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
            float alpha = max(0, min((-start.x) / ray_direction.x,
                                (far_corner.x - start.x) / ray_direction.x));
            start += alpha * ray_direction;
            alpha = max(0, min((-start.y) / ray_direction.y,
                           (far_corner.y - start.y) / ray_direction.y));
            start += alpha * ray_direction;
            alpha = max(0, min((-start.z) / ray_direction.z,
                           (far_corner.z - start.z) / ray_direction.z));
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
                float lambertian_intensity = dot(
                    normal, normalize(surface_to_light)
                ) / s2l_distance;
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
        }"""

        self.shader = Shader(
            self.render_pass,
            # An identifying name
            "A simple shader",
            vertex_shader,
            fragment_shader,
            blend_mode=Shader.BlendMode.AlphaBlend
        )

        light_pos = np.eye(4, 3).astype(np.float32)
        light_color = np.eye(4, 3).astype(np.float32)
        self.shader.set_buffer("scale_factor", np.array(1.0, dtype=np.float32))
        self.shader.set_buffer("light_pos[0]", light_pos.flatten())
        self.shader.set_buffer("light_color[0]", light_color.flatten())

        self.shader.set_buffer("indices", np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32))
        self.shader.set_buffer("position", np.array(
            [[-1, -1, 0],
             [1, -1, 0],
             [1, 1, 0],
             [-1, 1, 0]],
            dtype=np.float32
        ))
        self._texture = None
        self.get_dummy_texture()
        self.shader.set_texture3d("scalar_field", self._texture)
        self._camera_matrix = np.eye(4, dtype=np.float32)

    def mouse_button_event(self, p, button, down, modifiers):
        super(TestApp, self).mouse_button_event(p, button, down, modifiers)

        for tooltip in self._tooltips:
            if tooltip.contains(p):
                handled = tooltip.mouse_button_event(p, button, down, modifiers)
                if handled:
                    return True

        if self._handle_mouse_down:
            self._handle_mouse_down(p, button, down, modifiers)

        return True

    def get_dummy_texture(self):
        # cube
        buff = np.mgrid[0:1:196j, 0:1:196j, 0:1:196j].astype(np.float32)
        sdf = np.maximum(
            np.maximum(0.5 - buff[0, ...], -0.5 + buff[0, ...]),
            np.maximum(np.maximum(0.5 - buff[1, ...], -0.5 + buff[1, ...]),
                       np.maximum(0.5 - buff[2, ...], -0.5 + buff[2, ...]))
        ) - 0.25
        """
        # sphere
        buff = np.mgrid[-1:1:128j, -1:1:128j, -1:1:128j].astype(np.float32)
        sdf = np.linalg.norm(buff, 2, 0) - 0.5
        """
        """
        # armadillo
        sdf = np.load("/home/claudio/workspace/adventures-in-tensorflow/volume_armadillo.npz")
        sdf = (sdf['scalar_field'] - sdf['target_level']).astype(np.float32)
        #"""

        # Invert SDF sign if it is negative outside
        sdf = sdf * np.sign(sdf[0, 0, 0])

        self._texture = Texture3D(
            Texture.PixelFormat.R,
            Texture.ComponentFormat.Float32,
            sdf.shape,
            wrap_mode=Texture.WrapMode.ClampToEdge
        )
        self._texture.upload(sdf)
        self.shader.set_buffer("image_resolution", np.array(sdf.shape[0], dtype=np.float32))
        return self._texture

    def draw(self, ctx):
        super(TestApp, self).draw(ctx)
        for tooltip in self._tooltips:
            tooltip.draw()

    def resize_event(self, size):
        super(TestApp, self).resize_event(size)
        self._toolbar.set_size((size[0], self._toolbar.size()[1]))

        self.update_tooltip_positions()
        self.rt_pos_data = self.rt_position.download()

    def draw_contents(self):
        if self.shader is None:
            return
        self.render_pass.resize(self.framebuffer_size())

        #t = (glfw.getTime())*0 + 45 * np.pi / 180
        #self._camera_matrix = np.float32(
        #    Matrix4f.translate([0.5, 0.5, 0.5]) @ Matrix4f.rotate([0, 1, 0], t) @ Matrix4f.translate([-0.5, -0.5, -0.5])
        #)

        s = self.size()
        view_scale = Matrix4f.scale([1, s[0] / s[1], 1])
        with self.render_pass:
            mvp = view_scale
            self.shader.set_buffer("mvp", np.float32(mvp).T)
            self.shader.set_buffer("object2camera", self._camera_matrix.T)
            with self.shader:
                self.shader.draw_array(Shader.PrimitiveType.Triangle, 0, 6, True)

        # Initialize world position buffer
        if self.rt_pos_data is None:
            self.rt_pos_data = self.rt_position.download()

            # Initialize tooltip

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(TestApp, self).keyboard_event(key, scancode,
                                               action, modifiers):
            return True

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True

        self._ctrl_keys = {
            'forward': glfw.KEY_W,
            'left': glfw.KEY_A,
            'right': glfw.KEY_D,
            'back': glfw.KEY_S,
            'up': glfw.KEY_E,
            'down': glfw.KEY_Q,
        }
        self._move_vectors = {
            'forward': np.array([[0, 0, -1, 1]], dtype=np.float32).T,
            'back': np.array([[0, 0, 1, 1]], dtype=np.float32).T,
            'left': np.array([[-1, 0, 0, 1]], dtype=np.float32).T,
            'right': np.array([[1, 0, 0, 1]], dtype=np.float32).T,
            'up': np.array([[0, 1, 0, 1]], dtype=np.float32).T,
            'down': np.array([[0, -1, 0, 1]], dtype=np.float32).T,
        }

        kb_event_handled = False

        for movement, ctrl_key in self._ctrl_keys.items():
            if key == ctrl_key:
                move_matrix = np.concatenate(
                    [np.eye(4, 3, dtype=np.float32), 0.01*self._move_vectors[movement]], 1)
                move_matrix[3, 3] = 1
                self._camera_matrix = np.matmul(self._camera_matrix,
                                                move_matrix)
                kb_event_handled = True

        if kb_event_handled:
            self.update_tooltip_positions()
            self.rt_pos_data = self.rt_position.download()

        return kb_event_handled

    def scroll_event(self, p, rel):
        self._scale_power += rel[1]
        self.shader.set_buffer(
            "scale_factor", np.array(0.95**self._scale_power, dtype=np.float32))
        self.update_tooltip_positions()

    def mouse_motion_event(self, p, rel, button, modifiers):
        if super(TestApp, self).mouse_motion_event(p, rel, button, modifiers):
            return True

        for tooltip in self._tooltips:
            handled = tooltip.mouse_drag_event(p, rel, button, modifiers, self.rt_pos_data)
            if handled:
                return True

        mouse_event_handled = False

        if button == glfw.MOUSE_BUTTON_2:
            screen_size = np.max(self.size()) / 2
            new_fwd = np.array([rel.x/screen_size, -rel.y/screen_size, 1])
            new_fwd = new_fwd / np.linalg.norm(new_fwd)
            identity_up = np.array([0, 1, 0])
            new_left = np.cross(identity_up, new_fwd)
            new_up = -np.cross(new_left, new_fwd)
            rot_mat = np.concatenate([
                new_left[:, np.newaxis], new_up[:, np.newaxis], new_fwd[:, np.newaxis], np.zeros((3, 1))
            ], axis=1)
            rot_mat = np.concatenate([rot_mat, [[0, 0, 0, 1]]], 0).astype(np.float32)
            self._camera_matrix = np.matmul(
                self._camera_matrix,
                rot_mat
            )

            mouse_event_handled = True

        if mouse_event_handled:
            self.update_tooltip_positions()
            self.rt_pos_data = self.rt_position.download()

        return mouse_event_handled

    def update_tooltip_positions(self):
        # Create camera matrix
        window_size = self.size()
        focal_length = 1
        scale_factor = np.array(0.95**self._scale_power, dtype=np.float32)
        projection_matrix = np.array([
            [focal_length * window_size[0], 0, 0, 0],
            [0, focal_length * window_size[0], 0, 0],
            [0, 0, window_size[0], 0],
            [0, 0, -1, 0],
        ])
        focal_center = np.array([0.5, 0.5, focal_length, 0])
        # Transform from world space to camera space
        world2camera = np.linalg.inv(self._camera_matrix)
        # Create viewport matrix
        viewport_matrix = np.array([
            [1./scale_factor, 0, 0, 0.5 * window_size[0]],
            [0, -1./scale_factor, 0, 0.5 * window_size[1]],
        ])

        for tooltip in self._tooltips:
            # Bring tip coordinate from world to camera space
            tip_pos = world2camera @ tooltip.tip_position_world - focal_center
            # Perform camera projection and homogeneous normalization
            tip_pos = np.matmul(projection_matrix, tip_pos)
            tip_pos = tip_pos / tip_pos[-1]
            # Transform to screen coordinates
            tip_pos = viewport_matrix @ tip_pos
            tooltip.tip_position_screen = tip_pos.astype(np.int32)


def uithread():
    nanogui.init()
    test1 = TestApp()
    test1.draw_all()
    test1.set_visible(True)
    #test2 = TestApp()
    #test2.draw_all()
    #test2.set_visible(True)
    nanogui.mainloop(refresh=1 / 60.0 * 1000)
    del test1
    #del test2
    gc.collect()
    nanogui.shutdown()
    pass


if __name__ == "__main__":
    t = threading.Thread(target=uithread, daemon=False)
    t.start()
    q = queue.Queue()

    #while True:
    #    time.sleep(0.1)

    t.join()
