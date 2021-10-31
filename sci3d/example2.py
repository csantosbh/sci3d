#!/usr/bin/env python

import queue
import threading

import nanogui
import gc
import numpy as np
import icecream

from sci3d.tooltip import Tooltip

icecream.install()
ic.configureOutput(includeContext=True)

from nanogui import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D,\
    Matrix4f, glfw, icons


class Sci3DWindow(Screen):
    @property
    def depth_buffer(self):
        return self._rt_pos_data

    def __init__(self, size=(1024, 768), title='Sci3D'):
        super(Sci3DWindow, self).__init__(size, title)

        # Events
        self._handle_mouse_down = None

        # UI
        self._toolbar = None
        self._tooltips = []
        self._scale_power = 0

        self._setup_toolbar()
        self.perform_layout()

        self._rt_color = Texture(
            # TODO we dont need rgba
            Texture.PixelFormat.RGBA,
            Texture.ComponentFormat.UInt8,
            # TODO recreate texture when size changes
            self.size(),
            # TODO using shaderread has implications that I'm not sure I want
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self._rt_position = Texture(
            Texture.PixelFormat.RGB,
            Texture.ComponentFormat.Float32,
            # TODO recreate texture when size changes
            self.size(),
            # TODO using shaderread has implications that I'm not sure I want
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self._render_pass = RenderPass([self._rt_color, self._rt_position], blit_target=self)
        self._render_pass.set_clear_color(0, Color(0.3, 0.3, 0.32, 1.0))

        self._rt_pos_data = None

        # We currently only support opengl
        assert(nanogui.api == 'opengl')

        self._camera_matrix = np.eye(4, dtype=np.float32)
        self._plot_drawer = None

    def set_plot_drawer(self, plot_drawer):
        self._plot_drawer = plot_drawer

    def mouse_button_event(self, p, button, down, modifiers):
        super(Sci3DWindow, self).mouse_button_event(p, button, down, modifiers)

        for tooltip in self._tooltips:
            if tooltip.contains(p):
                handled = tooltip.mouse_button_event(p, button, down, modifiers)
                if handled:
                    return True

        if self._handle_mouse_down:
            self._handle_mouse_down(p, button, down, modifiers)

        return True

    def draw(self, ctx):
        super(Sci3DWindow, self).draw(ctx)

        for tooltip in self._tooltips:
            tooltip.draw()

    def resize_event(self, size):
        super(Sci3DWindow, self).resize_event(size)
        self._toolbar.set_size((size[0], self._toolbar.size()[1]))

        self._update_tooltip_positions()
        self._rt_pos_data = self._rt_position.download()

    def draw_contents(self):
        self._render_pass.resize(self.framebuffer_size())

        if self._plot_drawer:
            with self._render_pass:
                self._plot_drawer.draw()

        # Initialize world position buffer
        if self._rt_pos_data is None:
            self._rt_pos_data = self._rt_position.download()

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(Sci3DWindow, self).keyboard_event(key, scancode,
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
            self._update_tooltip_positions()
            self._rt_pos_data = self._rt_position.download()

        return kb_event_handled

    def scroll_event(self, p, rel):
        self._scale_power += rel[1]
        self._update_tooltip_positions()

    def mouse_motion_event(self, p, rel, button, modifiers):
        if super(Sci3DWindow, self).mouse_motion_event(p, rel, button, modifiers):
            return True

        for tooltip in self._tooltips:
            handled = tooltip.mouse_drag_event(p, rel, button, modifiers)
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
            self._update_tooltip_positions()
            self._rt_pos_data = self._rt_position.download()

        return mouse_event_handled

    def _update_tooltip_positions(self):
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
            tooltip.set_tip_position_screen_without_world_update(tip_pos.astype(np.int32))

    def _setup_toolbar(self):
        self._toolbar = Window(self, "")
        self._toolbar.set_position((0, 0))

        toolbar = Widget(self._toolbar)
        padding = 2
        layout = BoxLayout(Orientation.Horizontal, Alignment.Middle, padding, 6)
        toolbar.set_layout(layout)

        def create_tooltip(p, button, down, modifiers):
            if down:
                make_tooltip_btn.set_pushed(False)
                self._tooltips.append(Tooltip(
                    tip_position_screen=p,
                    parent_screen=self,
                    on_destroy_cbk=lambda tt: self._tooltips.remove(tt)
                ))
                self._handle_mouse_down = None

        def cbk_handler():
            self._handle_mouse_down = create_tooltip

        make_tooltip_btn = ToolButton(toolbar, icons.FA_CROSSHAIRS)
        make_tooltip_btn.set_callback(cbk_handler)

        self.perform_layout()
        btn_size = make_tooltip_btn.size()
        self._toolbar.set_size((2*padding + self.size()[0], 2*padding + btn_size[1]))


def uithread():
    nanogui.init()
    test1 = Sci3DWindow()
    test1.draw_all()
    test1.set_visible(True)
    #test2 = Sci3DWindow()
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
