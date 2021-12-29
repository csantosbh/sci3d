import abc
from sci3d.common import BoundingBox


class BasicSurface(abc.ABC):
    def __init__(self, window):
        self._window = window

    def set_title(self, title: str):
        """
        Set window title

        :param title: Window title
        """
        self._window.set_caption(title)

    @abc.abstractmethod
    def get_bounding_box(self) -> BoundingBox:
        """
        Return [width, height, depth]
        :return:
        """
        raise NotImplementedError
