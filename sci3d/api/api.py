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
    """
    Create a new empty window

    Any subsequent calls to plotting functions will use the newly created window.
    If this function is not called before a plotting function, the plot will be
    made on the current window (or an empty window will be created, if none exists yet).
    """
    _instantiate_window()


def isosurface(volume: np.ndarray,
               common_params: Params = Params()
               ) -> plottypes.isosurface.IsosurfaceApi:
    """
    Plot the 0-level set of a volumetric scalar field

    :param volume: Input volume. Must be a rank 3 tensor of shape [nz, ny, nx] and float32 type TODO confirm zyx order
    :param common_params: Control light, camera, transforms etc. See Params for details
    :return: Object that allows updating the plot
    """
    assert(volume.ndim == 3)

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
    """
    Plot triangular 3D mesh

    Face culling is enabled, and therefore faces rendered in counter-clockwise order are not visible.

    :param vertices: Rank 2 of shape [n_vertices, 3] and type float32
    :param triangles: Rank 2 of shape [n_triangles, 3] and type uint32
    :param normals: Rank 2 of shape [n_vertices, 3] and type float32
    :param colors: Rank 2 of shape [n_vertices, 3] and type float32
    :param common_params: Control light, camera, transforms etc. See Params for details.
    :return: Object that allows updating the plot
    """
    assert(vertices.ndim == 2)
    assert(vertices.shape[1] == 3)
    assert(vertices.dtype == np.float32)

    assert(triangles.ndim == 2)
    assert(triangles.shape[1] == 3)
    assert(triangles.dtype == np.uint32)

    if normals is not None:
        assert(normals.ndim == 2)
        assert(normals.shape[0] == vertices.shape[0])
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


def get_window_count() -> int:
    """
    Get number of windows currently opened

    This can be useful for looping while windows are open.
    :return: Number of windows open
    """
    return nanogui.get_visible_window_count()


def shutdown():
    """
    Request the application main loop to terminate

    Under the hood, this calls nanogui.leave
    """
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
