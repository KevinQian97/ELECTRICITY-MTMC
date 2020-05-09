from collections import defaultdict

import numpy as np
import torch

from ..utils import pack_tensor
from .base import Frame, ObjectType, Tracker
from .deep_sort import nn_matching
from .deep_sort.detection import Detection
from .deep_sort.tracker import Tracker as dsTracker

TRACK_STATES = ['Tentative', 'Confirmed']


class DeepSort(Tracker):

    def __init__(self, video_name, fps,min_iou = 0.1):
        super().__init__(video_name, fps)
        self.trackers = {}
        for obj_type in ObjectType:
            metric = nn_matching.NearestNeighborDistanceMetric(
                "cosine", 0.5, 5)
            self.trackers[obj_type] = dsTracker(
                metric, 1-min_iou, int(2 * fps))
        self.finished_tracks = []

    def group_instances(self, instances):
        grouped_instances = defaultdict(list)
        for obj_i in range(len(instances)):
            obj_type = ObjectType(instances.pred_classes[obj_i].item())
            bbox = instances.pred_boxes.tensor[obj_i]
            feature = instances.roi_features[obj_i].numpy().copy()
            detection = Detection(
                bbox, instances.scores[obj_i], feature, obj_i)
            grouped_instances[obj_type].append(detection)
        return grouped_instances

    def get_tracked_instances(self, instances):
        track_ids = torch.zeros((len(instances)), dtype=torch.int)
        states = torch.zeros((len(instances)), dtype=torch.int)
        track_boxes = torch.zeros((len(instances), 4))
        image_speeds = torch.zeros((len(instances), 2))
        for obj_type, tracker in self.trackers.items():
            for track in tracker.tracks:
                if track.time_since_update > 0:
                    continue
                obj_i = track.current_detection.obj_index
                track_ids[obj_i] = track.track_id
                states[obj_i] = track.state
                track_boxes[obj_i] = torch.as_tensor(
                    track.to_tlbr(), dtype=torch.float)
                speed = torch.as_tensor([
                    track.mean[4], track.mean[5] + track.mean[7] / 2])
                image_speeds[obj_i] = speed * self.fps
            for track in tracker.deleted_tracks:
                self.finished_tracks.append((obj_type, track.track_id))
        instances.track_ids = track_ids
        instances.track_states = states
        instances.track_boxes = track_boxes
        instances.image_speeds = image_speeds
        if len(instances) > 0:
            instances.finished_tracks = pack_tensor(
                torch.as_tensor(self.finished_tracks), len(instances))
            self.finished_tracks.clear()
        else:
            instances.finished_tracks = torch.zeros((0, 0, 0))
        return instances

    def track(self, frame):
        grouped_instances = self.group_instances(frame.instances)
        for obj_type, tracker in self.trackers.items():
            tracker = self.trackers[obj_type]
            tracker.predict()
            tracker.update(grouped_instances[obj_type])
        instances = self.get_tracked_instances(frame.instances)
        return Frame(frame.image_id, frame.image, instances)
