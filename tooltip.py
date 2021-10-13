import numpy as np
from nanogui import icons, Color


class Tooltip(object):
    def __init__(self,
                 tip_position_screen,
                 parent_screen,
                 on_destroy_cbk,
                 box_tip_distance=10,
                 tip_radius=4,
                 tip_color=Color(48, 48, 48, 192),
                 text_color=Color(255, 255, 255, 255),
                 margin=(5, 5),
                 close_icon_size=10):
        super(Tooltip, self).__init__()
        self._parent_screen = parent_screen

        # Geometry
        self._tip_position_screen = tip_position_screen
        self._tip_position_world = None
        self._size = np.zeros((2,))

        # Appearance
        self._margin = margin
        self._tip_color = tip_color
        self._text_color = text_color
        self._tip_radius = tip_radius

        # Events
        self._is_moving_tip = None
        self._is_moving_window = None

        # Close icon
        self._close_icon_size = close_icon_size
        self._close_icon = eval(f'u"\\u{icons.FA_TIMES_CIRCLE:04x}"')
        # [minx, miny, maxx, maxy]
        self._close_icon_bounds = None
        self._destroy_requested_cbks = [on_destroy_cbk]

        # Caption
        self._caption = None
        # [minx, miny, maxx, maxy]
        self._caption_bounds = None

        self._update_tip_position_world()

        self._position = tip_position_screen - [self._size[0] // 2,
                                                box_tip_distance+int(self._size[1])]

    @property
    def tip_position_world(self):
        return self._tip_position_world

    @property
    def tip_position_screen(self):
        return self._tip_position_screen

    def set_tip_position_screen_without_world_update(self, value):
        self._tip_position_screen = value

    def contains(self, position):
        return self._is_mouse_over_window(position) or self._is_mouse_over_tip(position)

    def mouse_button_event(self, p, button, down, modifiers):
        if self._is_mouse_over_close_icon(p):
            [cbk(self) for cbk in self._destroy_requested_cbks]
            return True
        elif down:
            self._is_moving_tip = self._is_mouse_over_tip(p)
            self._is_moving_window = self._is_mouse_over_window(p)
        else:
            self._is_moving_tip = self._is_moving_window = False

        return self._is_moving_tip or self._is_moving_window

    def mouse_drag_event(self, p, rel, button, modifiers):
        if self._is_moving_tip:
            self._tip_position_screen += rel
            # Update world position and label
            self._update_tip_position_world()
        elif self._is_moving_window:
            self._position = self._position + rel

        return self._is_moving_window or self._is_moving_tip

    def draw(self):
        nvg = self._parent_screen.nvg_context()

        # Draw data point
        nvg.BeginPath()
        nvg.Circle(
            self._tip_position_screen[0], self._tip_position_screen[1], self._tip_radius
        )
        box_center = self._position + self._size / 2
        tip_box_dir = np.array(box_center - self._tip_position_screen)
        line_tip_intersection = (
                tip_box_dir / np.linalg.norm(tip_box_dir) * self._tip_radius
        ).astype(np.int32)
        nvg.MoveTo(*box_center)
        nvg.LineTo(*(self._tip_position_screen + line_tip_intersection))
        nvg.StrokeColor(self._tip_color)
        nvg.StrokeWidth(2)

        nvg.Stroke()

        # Background
        nvg.BeginPath()
        nvg.Rect(*self._position, *self._size)
        nvg.FillColor(self._tip_color)
        nvg.Fill()

        # Draw text
        nvg.BeginPath()
        nvg.FillColor(self._text_color)
        curr_pos = self._position
        nvg.TextBox(
            curr_pos[0] - self._caption_bounds[0] + self._margin[0],
            curr_pos[1] - self._caption_bounds[1] + self._margin[1],
            1e3,
            self._caption
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

    def _set_caption(self, caption):
        nvg = self._parent_screen.nvg_context()

        # Get caption bounds
        self._caption = caption
        self._caption_bounds = nvg.TextBoxBounds(
            0.0, 0.0, 1e3, self._caption
        )

        # Get close icon bounds
        nvg.Save()
        nvg.FontFace("icons")
        nvg.FontSize(self._close_icon_size)
        close_icon_bounds = nvg.TextBounds(0, 0, self._close_icon)
        nvg.Restore()

        self._close_icon_bounds = [
            self._caption_bounds[2] - close_icon_bounds[0] + 2 * self._margin[0],
            -self._caption_bounds[1] + self._margin[1],
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
            int(self._caption_bounds[2] - self._caption_bounds[0]) + 3*self._margin[0] + close_icon_size[0],
            int(self._caption_bounds[3] - self._caption_bounds[1]) + 2*self._margin[1]
        ])

    def _update_tip_position_world(self):
        self._tip_position_world = np.concatenate([
            self._parent_screen.rt_pos_data[self._tip_position_screen[1],
            self._tip_position_screen[0], :],
            [1]
        ])
        self._set_caption(self._build_caption())

    def _build_caption(self):
        return (f'({self._tip_position_world[0]:.4f},'
                f' {self._tip_position_world[1]:.4f},'
                f' {self._tip_position_world[2]:.4f})')

    def _is_mouse_over_tip(self, position):
        return np.linalg.norm(position - self._tip_position_screen) <= self._tip_radius

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