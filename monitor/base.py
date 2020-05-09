import uuid
from enum import IntEnum, auto
from typing import List

from ..detector.base import Frame, ObjectType


class EventType(IntEnum):

    vehicle_turning_left = auto()
    vehicle_turning_right = auto()
    vehicle_u_turn = auto()
    vehicle_starting = auto()
    vehicle_stopping = auto()
    vehicle_reversing = auto()


class Event(object):

    def __init__(self, event_type, object_ids, video_name,
                 start_time, end_time, score):
        self.event_id = uuid.uuid4()
        self.event_type = event_type
        self.object_ids = tuple(object_ids)
        self.video_name = video_name
        self.start_time = start_time
        self.end_time = end_time
        self.score = score

    def key(self):
        return (self.event_type, self.object_ids)

    def __repr__(self):
        return '<%s: event_type=%s, object_ids=%s, start_time=%d, ' \
            'end_time=%d, score=%.3f>' % (
                repr(self.__class__)[8:-2], self.event_type.name, self.object_ids, self.start_time, self.end_time, self.score)

    def to_official(self):
        official = {
            'activity': self.event_type.name, 'activityID': int(self.event_id),
            'presenceConf': self.score, 'localization': {self.video_name: {
                str(self.start_time): 1, str(self.end_time): 0}}}
        return official


class Monitor(object):

    def __init__(self, video_name: str, fps: float, stride: int = 1):
        self.video_name = video_name
        self.fps = fps
        self.stride = stride

    def monit(self, frame: Frame) -> List[Event]:
        raise NotImplementedError

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.video_name)
