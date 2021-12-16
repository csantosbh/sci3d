from typing import Union, Callable, Type, Optional, List, Dict
import time

import numpy as np

import nanogui

from sci3d.uithread import _lock
import sci3d.plottypes as plottypes
from sci3d.plottypes import mesh, isosurface
from sci3d.window import Sci3DWindow


_api_types = Union[
    plottypes.isosurface.IsosurfaceApi,
    plottypes.mesh.MeshApi
]
_plot_types = Union[
    plottypes.isosurface.Isosurface,
    plottypes.mesh.MeshSurface,
]
_figures: List[Sci3DWindow] = []
_current_figure: Optional[Sci3DWindow] = None


def figure():
    _instantiate_window()


def isosurface(volume: np.ndarray,
               **kwargs) -> isosurface.IsosurfaceApi:
    api_object = _add_surface_to_window(
        _current_figure,
        plottypes.isosurface.IsosurfaceApi,
        plottypes.isosurface.Isosurface,
        dict(volume=volume)
    )
    api_object.set_title(kwargs.get('title', 'Sci3D'))

    return api_object


def mesh(vertices: np.ndarray,
         triangles: np.ndarray,
         normals: Optional[np.ndarray] = None,
         colors: Optional[np.ndarray] = None,
         pose: Optional[np.ndarray] = None,
         **kwargs) -> mesh.MeshApi:
    assert(vertices.ndim == 2)
    assert(vertices.shape[1] == 3)
    assert(vertices.dtype == np.float32)

    assert(triangles.ndim == 2)
    assert(triangles.shape[1] == 3)
    assert(triangles.dtype == np.uint32)

    if normals is not None:
        assert(normals.ndim == 2)
        assert(normals.shape[1] == 3)
        assert(normals.dtype == np.float32)

    if colors is not None:
        assert(colors.ndim == 2)
        assert(colors.shape[1] == 3)
        assert(colors.dtype == np.float32)

    if pose is not None:
        assert(pose.ndim == 2)
        assert(pose.shape == (4, 4))
        assert(pose.dtype == np.float32)

    api_object = _add_surface_to_window(
        _current_figure,
        plottypes.mesh.MeshApi,
        plottypes.mesh.MeshSurface,
        dict(vertices=vertices,
             triangles=triangles,
             normals=normals,
             colors=colors,
             pose=pose)
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
                           plottype_params: Dict[str, object]
                           ) -> _api_types:
    finished = False
    api_object = None

    # Create window if none exist
    if len(_figures) == 0:
        window = _instantiate_window()

    def async_impl():
        nonlocal api_object
        nonlocal window

        plottype_obj = plottype_ctor(_current_figure, **plottype_params)
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
