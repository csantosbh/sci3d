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


class Tooltip(Window):
    def __init__(self,
                 parent,
                 tip_position,
                 tip_label,
                 box_tip_distance=10,
                 tip_radius=4,
                 tip_color=Color(48, 48, 48, 255)):
        super(Tooltip, self).__init__(parent, "")
        self._mouse_drag_last = None
        self._tip_position = tip_position
        self._margin = (5, 5)
        self._tip_color = tip_color
        self._tip_radius = tip_radius

        theme = Theme(self.screen().nvg_context())
        theme.m_window_drop_shadow_size = 3
        self.set_theme(theme)
        self.caption = None
        self.caption_bounds = None
        self.set_caption(tip_label)

        curr_size = self.size()
        self.set_position(tip_position - Vector2i(curr_size[0]//2, box_tip_distance+int(curr_size[1])))

    def set_caption(self, caption):
        self.caption = caption
        self.caption_bounds = self.screen().nvg_context().TextBoxBounds(
            0.0, 0.0, 1e3, self.caption
        )
        caption_size = (
            int(self.caption_bounds[2] - self.caption_bounds[0]) + 2*self._margin[0],
            int(self.caption_bounds[3] - self.caption_bounds[1]) + 2*self._margin[1]
        )
        self.set_size(caption_size)
        pass

    def mouse_button_event(self, p, button, down, modifiers):
        self._mouse_drag_last = p

    def mouse_drag_event(self, p, rel, button, modifiers):
        dp = p - self._mouse_drag_last
        self._mouse_drag_last = p
        self.set_position(self.position() + dp)

    def draw(self, nvg):
        # Temporarily disable scissoring
        nvg.Save()
        nvg.Reset()

        # Draw data point
        nvg.BeginPath()
        nvg.Circle(
            self._tip_position[0], self._tip_position[1], self._tip_radius
        )
        box_center = self.position() + self.size() / 2
        tip_box_dir = np.array(box_center - self._tip_position)
        line_tip_intersection = (
                tip_box_dir / np.linalg.norm(tip_box_dir) * self._tip_radius
        ).astype(np.int32)
        nvg.MoveTo(*box_center)
        nvg.LineTo(*(self._tip_position + line_tip_intersection))
        nvg.StrokeColor(self._tip_color)
        nvg.StrokeWidth(2)

        # Restore scissoring
        nvg.Stroke()
        nvg.Restore()

        super(Tooltip, self).draw(nvg)

        # Draw text
        nvg.BeginPath()
        curr_pos = self.position()
        nvg.TextBox(
            curr_pos[0]-self.caption_bounds[0] + self._margin[0],
            curr_pos[1]-self.caption_bounds[1] + self._margin[1],
            1e3,
            self.caption
        )
        nvg.Stroke()


class TestApp(Screen):
    def __init__(self):
        super(TestApp, self).__init__((1024, 768), "NanoGUI Test")
        self.shader = None

        self.mytest = Tooltip(self, tip_position=Vector2i(500, 500), tip_label="(3.1415, 2.7100, 0.0000)")

        self.perform_layout()

        self.render_pass = RenderPass([self])
        self.render_pass.set_clear_color(0, Color(0.3, 0.3, 0.32, 1.0))

        # We currently only support opengl
        assert(nanogui.api == 'opengl')

        vertex_shader = """
        #version 330
        uniform mat4 mvp;
        in vec3 position;
        out vec2 uv;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            uv = (position.xy * 0.5) + 0.5;
        }"""

        fragment_shader = """
        #version 330
        //#define CAMERA_ORTHOGONAL
        in vec2 uv;
        out vec4 color;

        uniform sampler3D scalar_field;
        uniform mat4 mv;
        uniform float image_resolution;
        
        vec4 intersect_isocurve_0(vec3 start, vec3 ray_direction) {
            vec3 ray_pos = start;
            float step_size = 2.0 / image_resolution;

            float sdf;
            float was_ray_previously_outside = 1;
            
            for(int i = 0; i < image_resolution / 2; ++i) {
                sdf = texture(scalar_field, ray_pos).r;
                float is_ray_outside = float(sdf >= 0);
                float step_direction = mix(-1, 1, is_ray_outside);
                float ray_crossed_isocurve = float(
                    is_ray_outside != was_ray_previously_outside
                );
                step_size = mix(step_size, step_size * 0.5, ray_crossed_isocurve);
                was_ray_previously_outside = is_ray_outside;
                ray_pos += step_direction * ray_direction * step_size;
            }
            return vec4(ray_pos, sdf);
        }
        
        float derivative(vec3 pos, vec3 eps) {
            return texture(scalar_field, pos + eps).r -
                   texture(scalar_field, pos - eps).r;
        }
        
        vec3 gradient(vec3 pos) {
            float eps = 1.0 / image_resolution;
            
            float dx = derivative(pos, vec3(eps, 0, 0));
            float dy = derivative(pos, vec3(0, eps, 0));
            float dz = derivative(pos, vec3(0, 0, eps));

            return vec3(dx, dy, dz);
        }
        
        void main() {
            vec3 curr_pos_l = vec3(uv, 0.0);
            vec3 curr_pos_w = (mv * vec4(curr_pos_l, 1)).xyz;
            
            // Light
            vec3 light_pos = (mv * vec4(0.5, 0.5, 0, 1)).xyz;
            vec3 light_color = vec3(0.5, 0.2, 0.2);

            // Discard fragment if inside volume
            if (sign(texture(scalar_field, curr_pos_w.xyz).r) < 0)
                discard;
            
            // Setup camera
#if defined(CAMERA_ORTHOGONAL)
            // orthogonal
            vec4 curr_dir = mv * vec4(0, 0, 1, 0);
#else
            // perspective
            float focal_length = 1.0;
            vec4 curr_dir = mv * vec4(
                normalize(curr_pos_l - vec3(0.5, 0.5, -focal_length)),
                0
            );
#endif

            // Get surface geometry
            vec4 iso_intersection = intersect_isocurve_0(curr_pos_w, curr_dir.xyz); 
            vec3 surface_pos = iso_intersection.xyz;
            float sdf_value = iso_intersection.w;
            float eps = 1e-1;
            if (abs(iso_intersection.w) > eps) discard;
            vec3 normal = normalize(gradient(surface_pos));

            // Apply lighting
            vec3 surface_to_light = light_pos - surface_pos;
            float s2l_distance = length(surface_to_light);
            float lambertian_intensity = dot(
                normal, normalize(surface_to_light)
            ) / s2l_distance;
            color.rgb = lambertian_intensity * light_color;
        }"""

        self.shader = Shader(
            self.render_pass,
            # An identifying name
            "A simple shader",
            vertex_shader,
            fragment_shader
        )

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

    def get_dummy_texture(self):
        buff = np.mgrid[-1:1:32j, -1:1:32j, -1:1:32j].astype(np.float32)
        sdf = np.linalg.norm(buff, 2, 0) - 0.5
        """
        sdf = np.load("/home/claudio/workspace/adventures-in-tensorflow/volume_armadillo.npz")
        sdf = (sdf['scalar_field'] - sdf['target_level']).astype(np.float32)
        """

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

    def draw_contents(self):
        if self.shader is None:
            return
        self.render_pass.resize(self.framebuffer_size())

        s = self.size()
        view_scale = Matrix4f.scale([1, s[0] / s[1], 1]) if s[0] > s[1] \
            else Matrix4f.scale([s[1] / s[0], 1, 1])
        with self.render_pass:
            mvp = view_scale @ Matrix4f.rotate([0, 0, 1], 0)
            self.shader.set_buffer("mvp", np.float32(mvp).T)
            t = (glfw.getTime())
            mv = Matrix4f.translate([0.5, 0.5, 0.5]) @ Matrix4f.rotate([0, 1, 0], t) @ Matrix4f.translate([-0.5, -0.5, -0.5])
            self.shader.set_buffer("mv", np.float32(mv).T)
            with self.shader:
                self.shader.draw_array(Shader.PrimitiveType.Triangle, 0, 6, True)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(TestApp, self).keyboard_event(key, scancode,
                                               action, modifiers):
            return True
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True
        return False

    def mouse_motion_event(self, p, rel, button, modifiers):
        self.mytest.set_caption(f'({p[0]}, {p[1]})\nok')
        pass


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
