import time

import nanogui
from nanogui import glfw
import threading

from sci3d.plottypes.isosurface import Isosurface

from sci3d.example2 import *


_lock = threading.Lock()
_window_count = 0


def uithread():
    nanogui.init()
    nanogui.set_server_mode(True)

    nanogui.mainloop(refresh=1 / 60.0 * 1000)

    nanogui.shutdown()


def _instantiate_window(plottype_ctor, params):
    finished = False

    def _instantiate_window_impl():
        window = Sci3DWindow()
        window.set_plot_drawer(plottype_ctor(window, params))
        window.draw_all()
        window.set_visible(True)

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui.call_async(_instantiate_window_impl)
    while not finished:
        time.sleep(0.1)
    _lock.release()


def isosurface(surface):
    _instantiate_window(Isosurface, surface)


def get_window_count():
    return nanogui.get_window_count()


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


t = threading.Thread(target=uithread, daemon=True)
t.start()
