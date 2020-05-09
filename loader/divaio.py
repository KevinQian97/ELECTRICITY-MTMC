import numpy as np
import torch

from .base import Loader
from .diva_io.video import VideoReader


class DivaIO(Loader):

    def __init__(self, video_path, parent_dir=''):
        super().__init__(video_path, parent_dir)
        self.video = VideoReader(video_path, parent_dir)
        self.fps = float(self.video.fps)

    def read_iter(self, batch_size=1, limit=None, stride=1, start=0):
        self.video.reset()
        if start != 0:
            self.video.seek(start)
        images, image_ids = [], []
        for image_id, frame in enumerate(self.video.get_iter(
                limit // stride, stride)):
            image = torch.as_tensor(frame.numpy())
            images.append(image)
            image_ids.append(image_id)
            if len(images) == batch_size:
                yield images, image_ids
                images, image_ids = [], []
        if len(images) > 0:
            yield images, image_ids
