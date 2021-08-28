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
                    IntBox, RenderPass, Shader, Texture, Matrix4f, \
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
        // TODO
        //in vec2 uv_in;
        out vec2 uv;
        void main() {
            gl_Position = mvp * vec4(position, 1.0);
            // TODO
            //uv = uv_in;
            uv = (position.xy * 0.5) + 0.5;
        }"""

        fragment_shader = """
        #version 330
        in vec2 uv;
        out vec4 color;
        uniform sampler2D my_volume;
        void main() {
            float tex = texture2D(my_volume, uv).r;
            color.rgb = vec3(tex);
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
        self.shader.set_texture("my_volume", self._texture)

    def get_dummy_texture(self):
        buff = np.mgrid[0:1:128j, 0:1:128j][0].astype(np.float32)
        self._texture = Texture(
            Texture.PixelFormat.R,
            Texture.ComponentFormat.Float32,
            buff.shape
        )
        self._texture.upload(buff)
        return self._texture

    def draw(self, ctx):
        super(TestApp, self).draw(ctx)

    def draw_contents(self):
        if self.shader is None:
            return
        self.render_pass.resize(self.framebuffer_size())

        s = self.size()
        with self.render_pass:
            mvp = Matrix4f.scale([s[1] / float(s[0]) * 0.25, 0.25, 0.25]) @ \
                  Matrix4f.rotate([0, 0, 1], glfw.getTime())
            self.shader.set_buffer("mvp", np.float32(mvp).T)
            with self.shader:
                self.shader.draw_array(Shader.PrimitiveType.Triangle, 0, 6, True)

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(TestApp, self).keyboard_event(key, scancode,
                                               action, modifiers):
            return True
        ic(f'keyboard {key}, {action}, {modifiers}')
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
