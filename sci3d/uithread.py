from typing import Callable

import threading
import time

import nanogui_sci3d


_lock = threading.Lock()


def _ui_thread():
    nanogui_sci3d.init()
    nanogui_sci3d.set_server_mode(True)

    nanogui_sci3d.mainloop(refresh=1 / 60.0 * 1000)

    nanogui_sci3d.shutdown()


def run_in_ui_thread(functor):
    finished = False

    def _run_functor():
        functor()

        nonlocal finished
        finished = True

    _lock.acquire()
    nanogui_sci3d.call_async(_run_functor)
    while not finished:
        time.sleep(1e-4)
    _lock.release()


_ui_thread_obj = threading.Thread(target=_ui_thread, daemon=True)
_ui_thread_obj.start()
