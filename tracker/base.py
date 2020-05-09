from collections import namedtuple
from functools import partial

from ..detector.base import Frame, ObjectType


class Tracker(object):

    def __init__(self, video_name: str, fps: float):
        self.video_name = video_name
        self.fps = fps

    def track(self, instances: Frame) -> Frame:
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (self.__module__, self.__class__.__name__,
                             self.video_name)
