import json
import os.path as osp
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress

from .base import Monitor, ObjectType

Event = namedtuple('Event', [
    'video_id', 'frame_id', 'obj_type', 'track_id',
    'track'])


class MovementMonitor(Monitor):

    def __init__(self, video_name, fps, stride, video_id, camera_id, img_height,
                 img_width, min_region_overlap=0.5, gaussian_std=0.667,
                 min_length=0.3, min_score=0.7):
        super().__init__(video_name, fps, stride)
        self.video_id = video_id
        self.camera_id = camera_id
        self.min_region_overlap = min_region_overlap
        self.gaussian_std = gaussian_std * fps
        self.min_length = max(3, min_length * fps)
        self.min_score = min_score
        self.tracks = defaultdict(list)
        self.last_frame_id = -1

    def forward_tracks(self, instances, image_id):
        boxes = instances.track_boxes.numpy()
        for obj_i in range(len(instances)):
            track_id = instances.track_ids[obj_i].item()
            obj_type = instances.pred_classes[obj_i].item()
            x1, y1, x2, y2 = boxes[obj_i]
            self.tracks[(obj_type, track_id)].append(
                (image_id, x1, y1, x2, y2))
        self.last_frame_id = image_id

    def get_event(self, obj_type, track_id):
        key = (obj_type, track_id)
        raw_track = self.tracks.pop(key, None)
        if raw_track is None:
            return None
        frame_id = raw_track[-1][0] + 1
        event = Event(self.video_id, frame_id, obj_type,
                      track_id, raw_track)
        return event

    def monit(self, frame):
        self.forward_tracks(frame.instances, frame.image_id)
        finished_tracks = frame.instances.finished_tracks.reshape(
            (-1, 2)).numpy()
        events = []
        for obj_type, track_id in finished_tracks:
            event = self.get_event(obj_type, track_id)
            if event is None:
                continue
            events.append(event)
        return events

    def finish(self):
        events = []
        for obj_type, track_id in [*self.tracks.keys()]:
            event = self.get_event(obj_type, track_id)
            if event is None:
                continue
            events.append(event)
        return events
