#!/usr/bin/env python

import nanogui_sci3d
import numpy as np
from typing import List

from sci3d.tooltip import Tooltip
import sci3d.common as common
from sci3d.api.basicsurface import BasicSurface, Params

from nanogui_sci3d import Color, Screen, Window, BoxLayout, ToolButton, Widget, \
    Alignment, Orientation, RenderPass, Shader, Texture, Texture3D,\
    Matrix4f, glfw, icons


class Grid(object):
    def __init__(self, window):
        self._window = window

        vertices = np.concatenate([
            np.reshape(np.transpose(np.mgrid[-10:10:40j, 0:0:1j, -10:10:2j], [1, 2, 3, 0]), [-1, 3]),
            np.reshape(np.transpose(np.mgrid[-10:10:2j, 0:0:1j, -10:10:40j], [3, 2, 1, 0]), [-1, 3])
        ], 0).astype(np.float32)

        indices = np.arange(vertices.shape[0]).reshape((-1, 2)).astype(np.uint32)
        self._projection = common.get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=window.camera_fov,
            hw_ratio=window.size()[1] / window.size()[0],
            scale_factor=1,
        )
        self._grid = common.Wireframe(
            window._render_pass,
            vertices,
            indices,
            np.full_like(vertices, 0.2784), self._projection
        )

    def resize_event(self):
        self._projection = common.get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=self._window.camera_fov,
            hw_ratio=self._window.size()[1] / self._window.size()[0],
            scale_factor=1,
        )

    def draw(self):
        object2world = np.eye(4, dtype=np.float32)
        world2camera = self._window.world2camera()
        self._grid.get_material().shader.set_buffer("object2world", object2world.T)
        self._grid.draw(world2camera, self._projection)


class Gizmo(object):
    def _make_axis(self,
                   rotation: np.ndarray,
                   color: np.ndarray):
        height = 0.005
        base = 0.0015
        vertices, triangles = common.cone(base, height, 5)

        upsidedown_r = common.rot_x(np.pi)
        upsidedown_t = np.array([[0, -height, 0]], dtype=np.float32)

        vertices = ((vertices + upsidedown_t) @ upsidedown_r.T) @ rotation.T

        colors = np.repeat([color], vertices.shape[0], axis=0).astype(np.float32)
        mesh = common.Mesh(
            self._window.render_pass,
            vertices, triangles, None, colors,
            self._projection
        )

        # Basic lighting for gizmos
        light_pos = np.zeros((4, 3)).astype(np.float32)
        light_color = np.zeros((4, 3)).astype(np.float32)
        light_color[0, :] = 1

        material = mesh.get_material()
        material.shader.set_buffer("light_pos[0]", light_pos.flatten())
        material.shader.set_buffer("light_color[0]", light_color.flatten())
        return mesh

    def __init__(self, window):
        self._window: Sci3DWindow = window
        self._projection = common.get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=window.camera_fov,
            hw_ratio=window.size()[1] / window.size()[0],
            scale_factor=1,
        )

        # TODO batch meshes together to avoid unnecessary draw calls
        self._meshes = [
            self._make_axis(common.rot_z(-np.pi/2), [1, 0, 0]),
            self._make_axis(np.eye(3, dtype=np.float32), [0, 1, 0]),
            self._make_axis(common.rot_x(np.pi / 2), [0, 0, 1]),
        ]

    def resize_event(self):
        self._projection = common.get_projection_matrix(
            near=0.1,
            far=1e3,
            fov=self._window.camera_fov,
            hw_ratio=self._window.size()[1] / self._window.size()[0],
            scale_factor=1,
        )

    def draw(self):
        width, height = self._window.size()
        depth = 0.2
        x = np.tan(0.9 * self._window.camera_fov / 2) * depth
        y = x * height / width
        object2world = common.forward_affine(
            self._window._camera_rotation.T,
            np.array([[-x, -y, -depth]], dtype=np.float32).T
        )
        world2camera = np.eye(4, dtype=np.float32)
        [m.get_material().shader.set_buffer("object2world", object2world.T) for m in self._meshes]
        [m.draw(world2camera, self._projection) for m in self._meshes]


class Sci3DWindow(Screen):
    @property
    def depth_buffer(self):
        return self._rt_pos_data

    @property
    def camera_fov(self):
        return self._camera_fov

    @camera_fov.setter
    def camera_fov(self, value):
        self._camera_fov = np.array(value, dtype=np.float32)

    @property
    def scale_factor(self):
        return np.array(0.95 ** self._scale_power, dtype=np.float32)

    @property
    def render_pass(self):
        return self._render_pass

    def __init__(self, size=(1024, 768), title='Sci3D'):
        super(Sci3DWindow, self).__init__(size, title)

        # Camera
        self._camera_rotation = np.eye(3, dtype=np.float32)
        self._camera_position = np.zeros((3, 1), dtype=np.float32)
        self._camera_fov = np.array(45 * np.pi / 180, dtype=np.float32)
        self._plot_drawers: List[BasicSurface] = []

        # Events
        self._handle_mouse_down = None

        # UI
        self._toolbar = None
        self._tooltips = []
        self._scale_power = 0

        self._setup_toolbar()
        self.perform_layout()

        self._rt_color = Texture(
            Texture.PixelFormat.RGB,
            Texture.ComponentFormat.UInt8,
            # TODO recreate texture when size changes
            self.size(),
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self._rt_position = Texture(
            Texture.PixelFormat.RGB,
            Texture.ComponentFormat.Float32,
            # TODO recreate texture when size changes
            self.size(),
            flags=Texture.TextureFlags.ShaderRead | Texture.TextureFlags.RenderTarget
        )
        self._rt_depth = Texture(
            Texture.PixelFormat.Depth,
            Texture.ComponentFormat.Float32,
            self.size(),
            flags=Texture.TextureFlags.RenderTarget
        )
        self._render_pass = RenderPass(
            [self._rt_color, self._rt_position], self._rt_depth, blit_target=self
        )
        self._render_pass.set_clear_color(0, Color(0.2353, 0.2353, 0.2353, 1.0))

        self._rt_pos_data = None

        # Additional UI
        self._gizmo = Gizmo(self)
        self._grid = Grid(self)

        # We currently only support opengl
        assert(nanogui_sci3d.api == 'opengl')

    def add_plot_drawer(self,
                        plot_drawer: BasicSurface,
                        common_params: Params):
        self._plot_drawers.append(plot_drawer)

        override_camera_pose = (
            common_params.camera_position is not None or
            common_params.camera_rotation is not None
        )
        if override_camera_pose:
            if common_params.camera_position is not None:
                self._camera_position = common_params.camera_position
            if common_params.camera_rotation is not None:
                self._camera_rotation = common_params.camera_rotation
        elif common_params.reset_camera:
            self._reset_camera()

    def world2camera(self):
        return common.inverse_affine(self._camera_rotation, self._camera_position)

    def camera2world(self):
        return common.forward_affine(self._camera_rotation, self._camera_position)

    def draw(self, ctx):
        super(Sci3DWindow, self).draw(ctx)

        self._update_camera_position()

        for tooltip in self._tooltips:
            tooltip.draw()

    def draw_contents(self):
        self._render_pass.resize(self.framebuffer_size())

        with self._render_pass:
            for plot_drawer in self._plot_drawers:
                plot_drawer.draw()

            self._gizmo.draw()
            self._grid.draw()

        # Initialize world position buffer
        if self._rt_pos_data is None:
            self._rt_pos_data = self._rt_position.download()

    def mouse_button_event(self, p, button, down, modifiers):
        super(Sci3DWindow, self).mouse_button_event(p, button, down, modifiers)

        for tooltip in self._tooltips:
            if tooltip.contains(p):
                handled = tooltip.mouse_button_event(p, button, down, modifiers)
                if handled:
                    return True

        if self._handle_mouse_down:
            self._handle_mouse_down(p, button, down, modifiers)

        if not down:
            # TODO we can store depth only and estimate XY from it
            self._rt_pos_data = self._rt_position.download()

        return True

    def resize_event(self, size):
        super(Sci3DWindow, self).resize_event(size)
        self._toolbar.set_size((size[0], self._toolbar.size()[1]))

        self._update_tooltip_positions()
        self._rt_pos_data = self._rt_position.download()
        self._grid.resize_event()
        self._gizmo.resize_event()

    def keyboard_event(self, key, scancode, action, modifiers):
        if super(Sci3DWindow, self).keyboard_event(key, scancode,
                                                   action, modifiers):
            return True

        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            self.set_visible(False)
            return True

        if action == glfw.RELEASE:
            self._rt_pos_data = self._rt_position.download()

        if key == glfw.KEY_F and action == glfw.PRESS:
            self._reset_camera()

        return False

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

            identity_up = np.array([0, 1, 0], dtype=np.float32)

            new_fwd = np.array([rel.x/screen_size, -rel.y/screen_size, 1], dtype=np.float32)
            new_fwd = new_fwd / np.linalg.norm(new_fwd)

            new_left = np.cross(identity_up, new_fwd)

            new_up = -np.cross(new_left, new_fwd)

            self._camera_rotation = self._camera_rotation @ np.concatenate([
                new_left[:, np.newaxis], new_up[:, np.newaxis], new_fwd[:, np.newaxis]
            ], axis=1)
            self._camera_rotation = common.orthonormalize(self._camera_rotation)

            self._update_tooltip_positions()

        return mouse_event_handled

    def _reset_camera(self):
        if len(self._plot_drawers) > 0:
            full_bbox = self._plot_drawers[0].get_bounding_box()
            for drawer in self._plot_drawers[1:]:
                full_bbox = full_bbox.union(drawer.get_bounding_box())

            # Position camera in front of full_bbox
            camera_xy = (
                                full_bbox.lower_bound[0:2] + full_bbox.upper_bound[0:2]
                        ) / 2
            distance_factor = 0.6
            z_offset = 0.5 * full_bbox.height/np.tan(distance_factor * self.camera_fov/2)
            self._camera_position = np.concatenate([
                camera_xy, full_bbox.upper_bound[2:3] + z_offset
            ])[..., np.newaxis]
        else:
            self._camera_position = np.zeros((3, 1), dtype=np.float32)

        self._scale_power = 0
        self._camera_rotation = np.eye(3, dtype=np.float32)

    def _update_camera_position(self):
        ctrl_keys = {
            'forward': glfw.KEY_W,
            'left': glfw.KEY_A,
            'right': glfw.KEY_D,
            'back': glfw.KEY_S,
            'up': glfw.KEY_E,
            'down': glfw.KEY_Q,
        }

        move_vectors = {
            'forward': -self._camera_rotation[:, 2:3],
            'back': self._camera_rotation[:, 2:3],
            'left': -self._camera_rotation[:, 0:1],
            'right': self._camera_rotation[:, 0:1],
            'up': self._camera_rotation[:, 1:2],
            'down': -self._camera_rotation[:, 1:2],
        }

        kb_event_handled = False

        movement_direction = np.zeros((3, 1), dtype=np.float32)
        for movement, ctrl_key in ctrl_keys.items():
            if self.get_key_status(ctrl_key) == glfw.PRESS:
                movement_direction += move_vectors[movement]
                kb_event_handled = True

        speed = 0.01
        if self.get_key_status(glfw.KEY_LEFT_SHIFT) == glfw.PRESS:
            speed *= 10

        if kb_event_handled:
            eps = 1e-6
            movement_direction = movement_direction / (eps + np.linalg.norm(movement_direction))
            self._camera_position += speed * movement_direction
            self._update_tooltip_positions()

    def _update_tooltip_positions(self):
        # Create camera matrix
        width, height = self.size()
        projection_matrix = common.get_projection_matrix(
            0.1, 1e3, self.camera_fov,
            height / width, self.scale_factor
        )
        # Create viewport matrix
        viewport_matrix = np.array([
            [0.5 * width, 0, 0, 0.5 * width],
            [0, -0.5 * height, 0, 0.5 * height],
        ])

        for tooltip in self._tooltips:
            # Bring tip coordinate from world to camera space
            tip_pos = projection_matrix @ self.world2camera() @ tooltip.tip_position_world
            # Perform homogeneous normalization
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
