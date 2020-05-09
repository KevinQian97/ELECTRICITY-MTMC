import time
import torch
from collections import namedtuple

from ..pipeline import Stage, LoggedTask


DetectorResource = namedtuple('DetectorResource', ['gpu_id'])


class DetectorStage(Stage):

    def __init__(self, detector_class, gpus=None, n_detector_per_gpu=1,
                 **kwargs):
        super().__init__()
        self.detector_class = detector_class
        if gpus is not None:
            self.gpus = gpus
            self.n_detector_per_gpu = n_detector_per_gpu
        else:
            self.gpus = [None]
            self.n_detector_per_gpu = 1
        self.n_detector = len(self.gpus) * n_detector_per_gpu
        self.gpu_i = 0
        self.detector = None
        self.kwargs = kwargs

    def pipeline_init_fn(self):
        gpu_id = self.gpus[self.gpu_i]
        res = DetectorResource(gpu_id)
        self.gpu_i = (self.gpu_i + 1) % len(self.gpus)
        return res

    def get_worker_num(self):
        return self.n_detector

    def pipeline_fn(self, res, task):
        super().pipeline_fn(res, task)
        try:
            assert isinstance(task, LoggedTask)
            if task.meta.get('end', False):
                yield_task = LoggedTask(
                    None, prev_task=task, start_time=time.time(),
                    processor=self.detector)
                yield yield_task
                return
            assert isinstance(task.value, list)
            assert isinstance(task.value[0], torch.Tensor)
            if self.detector is None:
                self.detector = self.detector_class(
                    gpu_id=res.gpu_id, **self.kwargs)
                self.logger.debug('Started detector: %s', self.detector)
            images = task.value
            image_ids = task.meta.pop('image_ids')
            start_time = time.time()
            frames = self.detector.detect(images, image_ids)
            for frame in frames:
                yield_task = LoggedTask(
                    frame, prev_task=task, start_time=start_time,
                    processor=self.detector)
                yield_task.meta['image_id'] = frame.image_id
                yield yield_task
        except Exception as e:
            if self.detector is None:
                raise e
            self.logger.exception('Task failed: %s', task)
