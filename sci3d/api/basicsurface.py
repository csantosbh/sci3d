import abc
from typing import Optional

import numpy as np
from dataclasses import dataclass

import sci3d.common as common
import sci3d.materials as materials
from sci3d.uithread import run_in_ui_thread


@dataclass
class Params:
    """
    Common scene parameters for plots
    """
    window_title: str = 'Sci3D'
    reset_camera: bool = True
    # Shape [3, 1]
    object_position: np.ndarray = np.zeros((3, 1), dtype=np.float32)
    # Shape [3, 3]
    object_rotation: np.ndarray = np.eye(3, dtype=np.float32)
    # Shape [3, 1]
    camera_position: Optional[np.ndarray] = None
    # Shape [3, 3] rotation matrix
    camera_rotation: Optional[np.ndarray] = None


class BasicSurface(abc.ABC):
    def __init__(self, window, common_params: Params):
        self._window = window
        self._object_position = common_params.object_position
        self._object_rotation = common_params.object_rotation
        self._num_lights = 4

    @abc.abstractmethod
    def get_bounding_box(self) -> common.BoundingBox:
        """
        Return [width, height, depth]
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_material(self) -> materials.Material:
        raise NotImplementedError

    def post_init(self):
        # Setup basic lights
        light_pos = np.eye(4, 3).astype(np.float32)
        light_color = np.eye(4, 3).astype(np.float32)
        self.set_lights(light_pos, light_color)

    def set_transform(self,
                      position: np.ndarray,
                      rotation: np.ndarray):
        self._object_position = position
        self._object_rotation = rotation

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if light_pos.shape[0] != self._num_lights:
            light_pos = np.pad(light_pos, [[0, self._num_lights - light_pos.shape[0]], [0, 0]])

        if light_color.shape[0] != self._num_lights:
            light_color = np.pad(light_color, [[0, self._num_lights - light_color.shape[0]], [0, 0]])

        material = self.get_material()
        material.set_uniform("light_pos[0]", light_pos.flatten())
        material.set_uniform("light_color[0]", light_color.flatten())


class BasicSurfaceApi(abc.ABC):
    def __init__(self, window, plot_drawer: BasicSurface):
        self._window = window
        self._plot_drawer = plot_drawer

    def set_title(self, title):
        self._window.set_caption(title)

    def set_transform(self, position, rotation):
        self._plot_drawer.set_transform(position, rotation)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        if not self._window.visible():
            return

        def impl():
            self._plot_drawer.set_lights(light_pos, light_color)

        run_in_ui_thread(impl)
