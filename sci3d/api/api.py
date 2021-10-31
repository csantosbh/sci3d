from typing import Union, Callable, Type
import time

import numpy as np

import nanogui

from sci3d.uithread import _lock
import sci3d.plottypes as plottypes
import sci3d.plottypes.isosurface as isosurface
from sci3d.window import Sci3DWindow


_api_types = Union[
    isosurface.IsosurfaceApi,
]
_plot_types = Union[
    isosurface.Isosurface,
]


def isosurface(volume: np.ndarray,
               **kwargs) -> plottypes.isosurface.IsosurfaceApi:
    api_object = _instantiate_window(
        plottypes.isosurface.IsosurfaceApi, plottypes.isosurface.Isosurface, volume
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
        time.sleep(0.1)
    _lock.release()


def _instantiate_window(api_ctor: Type[_api_types],
                        plottype_ctor: Type[_plot_types],
                        *plottype_params
                        ) -> _api_types:
    finished = False
    api_object = None

    def _instantiate_window_impl():
        window = Sci3DWindow()
        window.set_plot_drawer(plottype_ctor(window, *plottype_params))
        window.draw_all()
        window.set_visible(True)

        nonlocal api_object
        api_object = api_ctor(window)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(_instantiate_window_impl)
    while not finished:
        time.sleep(0.1)
    _lock.release()

    return api_object
