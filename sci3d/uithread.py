from typing import Callable

import threading
import time

import nanogui


_lock = threading.Lock()


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
        time.sleep(1e-4)
    _lock.release()


_ui_thread_obj = threading.Thread(target=_ui_thread, daemon=True)
_ui_thread_obj.start()
