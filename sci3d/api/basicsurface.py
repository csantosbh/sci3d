import abc
from typing import Optional

import numpy as np
from dataclasses import dataclass

from sci3d.common import BoundingBox
from sci3d.uithread import run_in_ui_thread


@dataclass
class Params:
    window_title: str = 'Sci3D'
    reset_camera: bool = True
    # Shape [3, 1]
    camera_position: Optional[np.ndarray] = None
    # Shape [3, 3] rotation matrix
    camera_rotation: Optional[np.ndarray] = None


class BasicSurface(abc.ABC):
    def __init__(self, window):
        self._window = window

    @abc.abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """
        Return [width, height, depth]
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        raise NotImplementedError


class BasicSurfaceApi(abc.ABC):
    def __init__(self, window, plot_drawer: BasicSurface):
        self._window = window
        self._plot_drawer = plot_drawer

    def set_title(self, title):
        self._window.set_caption(title)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if not self._window.visible():
            return

        def impl():
            self._plot_drawer.set_lights(light_pos, light_color)

        run_in_ui_thread(impl)
