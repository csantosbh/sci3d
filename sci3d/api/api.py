from typing import Union, Callable, Type, Optional, List
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
_figures: List[Sci3DWindow] = []
_current_figure: Optional[Sci3DWindow] = None


def figure():
    _instantiate_window()


def isosurface(volume: np.ndarray,
               **kwargs) -> plottypes.isosurface.IsosurfaceApi:
    api_object = _add_surface_to_window(
        _current_figure,
        plottypes.isosurface.IsosurfaceApi,
        plottypes.isosurface.Isosurface,
        volume
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


def _instantiate_window() -> Sci3DWindow:
    global _current_figure
    global _figures

    finished = False
    new_window: Optional[Sci3DWindow] = None

    def async_impl():
        nonlocal new_window

        new_window = Sci3DWindow()
        new_window.draw_all()
        new_window.set_visible(True)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(async_impl)

    while not finished:
        time.sleep(0.1)

    _current_figure = new_window
    _figures.append(new_window)
    _lock.release()

    return new_window


def _add_surface_to_window(window: Sci3DWindow,
                           api_ctor: Optional[Type[_api_types]],
                           plottype_ctor: Optional[Type[_plot_types]],
                           *plottype_params
                           ) -> _api_types:
    finished = False
    api_object = None

    # Create window if none exist
    if len(_figures) == 0:
        window = _instantiate_window()

    def async_impl():
        nonlocal api_object
        nonlocal window

        plottype_obj = plottype_ctor(_current_figure, *plottype_params)
        _current_figure.add_plot_drawer(plottype_obj)
        api_object = api_ctor(_current_figure, plottype_obj)

        window.set_visible(True)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(async_impl)
    while not finished:
        time.sleep(0.1)
    _lock.release()

    return api_object
