from sci3d.plottypes.isosurface import Isosurface
from sci3d.example2 import Sci3DWindow


class IsosurfaceApi(object):
    def __init__(self, window: Sci3DWindow):
        self._window = window

    def set_isosurface(self, volume):
        from sci3d import run_in_ui_thread

        def set_isosurface_impl():
            if not self._window.visible():
                return

            isosurface_drawer: Isosurface = self._window._plot_drawer
            isosurface_drawer.set_isosurface(volume)

        run_in_ui_thread(set_isosurface_impl)

    def set_title(self, title):
        self._window.set_caption(title)
