import time
from dataclasses import dataclass
from typing import Union, Type, Optional, List, Dict

import nanogui
import numpy as np

import sci3d.plottypes as plottypes
import sci3d.plottypes.isosurface
import sci3d.plottypes.mesh
from sci3d.api.basicsurface import BasicSurfaceApi, Params
from sci3d.uithread import _lock
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
_current_window: Optional[Sci3DWindow] = None


def figure():
    _instantiate_window()


def isosurface(volume: np.ndarray,
               common_params: Params = Params()
               ) -> plottypes.isosurface.IsosurfaceApi:
    api_object = _add_surface_to_window(
        _current_window,
        plottypes.isosurface.IsosurfaceApi,
        plottypes.isosurface.Isosurface,
        dict(volume=volume),
        common_params,
    )

    return api_object


def mesh(vertices: np.ndarray,
         triangles: np.ndarray,
         normals: Optional[np.ndarray] = None,
         colors: Optional[np.ndarray] = None,
         common_params: Params = Params()
         ) -> plottypes.mesh.MeshApi:
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

    api_object = _add_surface_to_window(
        _current_window,
        plottypes.mesh.MeshApi,
        plottypes.mesh.MeshSurface,
        dict(vertices=vertices,
             triangles=triangles,
             normals=normals,
             colors=colors),
        common_params
    )

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
    global _current_window
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

    _current_window = new_window
    _figures.append(new_window)
    _lock.release()

    return new_window


def _add_surface_to_window(target_window: Sci3DWindow,
                           api_ctor: Optional[Type[_api_types]],
                           plottype_ctor: Optional[Type[_plot_types]],
                           plottype_params: Dict[str, object],
                           common_params: Params,
                           ) -> _api_types:
    finished = False
    api_object: Optional[BasicSurfaceApi] = None

    # Create window if none exist
    if len(_figures) == 0:
        target_window = _instantiate_window()

    def async_impl():
        nonlocal api_object
        nonlocal target_window

        plottype_obj = plottype_ctor(target_window, common_params, **plottype_params)
        target_window.add_plot_drawer(plottype_obj, common_params)
        api_object = api_ctor(target_window, plottype_obj)

        target_window.set_visible(True)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(async_impl)
    while not finished:
        time.sleep(0.1)
    _lock.release()

    api_object.set_title(common_params.window_title)

    return api_object
