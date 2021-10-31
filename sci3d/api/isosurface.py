import numpy as np

from sci3d.plottypes.isosurface import Isosurface
from sci3d.example2 import Sci3DWindow


class IsosurfaceApi(object):
    def __init__(self, window: Sci3DWindow):
        self._window = window

    def set_isosurface(self, volume):
        from sci3d import run_in_ui_thread

        if not self._window.visible():
            return

        def impl():
            isosurface_drawer: Isosurface = self._window._plot_drawer
            isosurface_drawer.set_isosurface(volume)

        run_in_ui_thread(impl)

    def set_title(self, title):
        self._window.set_caption(title)

    def set_lights(self, light_pos: np.ndarray, light_color: np.ndarray):
        from sci3d import run_in_ui_thread

        if not self._window.visible():
            return

        def impl():
            isosurface_drawer: Isosurface = self._window._plot_drawer
            isosurface_drawer.set_lights(light_pos, light_color)

        run_in_ui_thread(impl)
