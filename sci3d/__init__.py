import time as _time
from typing import Union, Callable

import nanogui as _nanogui
import threading as _threading

import sci3d.plottypes as _plottypes
import sci3d.api as _api

from sci3d.example2 import *


_lock = threading.Lock()
_window_count = 0
_api_types = Union[
    _api.isosurface.IsosurfaceApi,
]


def _ui_thread():
    nanogui.init()
    nanogui.set_server_mode(True)

    nanogui.mainloop(refresh=1 / 60.0 * 1000)

    nanogui.shutdown()


def run_in_ui_thread(functor):
    finished = False

    def _run_functor():
        functor()

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(_run_functor)
    while not finished:
        _time.sleep(1e-4)
    _lock.release()


def _instantiate_window(api_ctor: Callable[[Sci3DWindow], _api_types],
                        plottype_ctor,
                        params
                        ) -> _api_types:
    finished = False
    api_object = None

    def _instantiate_window_impl():
        window = Sci3DWindow()
        window.set_plot_drawer(plottype_ctor(window, params))
        window.draw_all()
        window.set_visible(True)

        nonlocal api_object
        api_object = api_ctor(window)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(_instantiate_window_impl)
    while not finished:
        _time.sleep(0.1)
    _lock.release()

    return api_object


def isosurface(volume: np.ndarray,
               **kwargs) -> _api.isosurface.IsosurfaceApi:
    api_object = _instantiate_window(
        _api.isosurface.IsosurfaceApi, _plottypes.isosurface.Isosurface, volume
    )
    api_object.set_title(kwargs.get('title', 'Sci3D'))

    return api_object


def get_window_count():
    return nanogui.get_visible_window_count()


def shutdown():
    finished = False

    def _shutdown_impl():
        nanogui.leave()

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(_shutdown_impl)
    while not finished:
        _time.sleep(0.1)
    _lock.release()


_ui_thread_obj = threading.Thread(target=_ui_thread, daemon=True)
_ui_thread_obj.start()
