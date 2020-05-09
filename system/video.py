import json
import os
import sys
import time
from collections import namedtuple

import GPUtil

from ..detector import DetectorStage
from ..detector.base import Frame
from ..loader import LoaderStage
from ..loader.stage import LoaderTask
from ..monitor import MinotorStage
from ..monitor.stage import MonitorResult
from ..pipeline import LoggedTask
from ..tracker import TrackerStage
from ..utils import progressbar
from .base import System
from .output import get_output
from .storage import Loader, Saver

VideoJob = namedtuple('Job', [
    'video_name', 'video_id', 'camera_id', 'n_frames'], defaults=[None])


class VideoSystem(System):

    def __init__(self, video_dir, cache_dir=None, stage_n=None,
                 n_gpu=4, batch_size=6, stride=1):
        available_gpus = GPUtil.getAvailable(
            limit=n_gpu, maxLoad=0.2, maxMemory=0.2)
        n_available_gpu = len(available_gpus)
        stages = [
            LoaderStage(),
            DetectorStage(available_gpus, 2),
            TrackerStage(),
            MinotorStage(cache_dir is not None)
        ]
        stages = stages[:stage_n]
        super().__init__(stages, video_dir, cache_dir, batch_size, stride)
        if n_available_gpu == 0:
            self.logger.warn('No gpus available, running on cpu')
        elif n_available_gpu < n_gpu:
            self.logger.warn(
                '%d gpus requested, but only %d gpus (gpu id: %s) available',
                n_gpu, n_available_gpu, available_gpus)
        else:
            self.logger.info(
                'Running on %d gpus (gpu id: %s)', n_gpu, available_gpus)
        if isinstance(stages[-1], (DetectorStage.func)) and n_gpu > 1:
            self.logger.warn(
                'Last stage is %s with %d gpus, '
                'results may be out of order and incomplete',
                stages[-1].__class__.__name__, n_gpu)
        self.videos_processed = []
        self.events = []

    def init(self, job, **kwargs):
        assert isinstance(job, VideoJob)
        n_image = self.batch_size
        if len(self.stages) > 1:
            n_image *= self.stages[1].n_detector
        loader_task = LoaderTask(
            job.video_name, self.video_dir, job.video_id, job.camera_id,
            limit=n_image, batch_size=self.batch_size, stride=self.stride)
        task = LoggedTask(
            loader_task, meta={}, start_time=time.time())
        super().init(task, **kwargs)

    def _save_filter(self, results_gen):
        for task in results_gen:
            if isinstance(task.value, MonitorResult):
                frame = task.value.frame
            else:
                frame = task.value
            if frame is None:
                continue
            instances = frame.instances
            instances.remove('pred_masks')
            frame = Frame(frame.image_id, None, instances)
            task.value = frame
            yield task

    def _log_events(self, results_gen, events):
        for task in results_gen:
            if hasattr(task.value, 'events') and task.value.events is not None:
                events.extend(task.value.events)
            yield task

    def do_job(self, job, print_result=0, **kwargs):
        assert isinstance(job, VideoJob)
        if not self.pipeline.inited:
            self.init(job)
        self.videos_processed.append(job.video_name)
        if job.n_frames is not None:
            limit = job.n_frames // self.stride
        else:
            limit = None
        loader_task = LoaderTask(
            job.video_name, self.video_dir, job.video_id, job.camera_id,
            limit=limit, batch_size=self.batch_size, stride=self.stride)
        task = LoggedTask(
            loader_task, meta={}, start_time=time.time())
        results_gen = super().process(task, **kwargs)
        events = []
        results_gen = self._log_events(results_gen, events)
        if self.cache_dir is not None:
            saver = Saver(job.video_name, self.cache_dir)
            results_gen = saver.save(self._save_filter(results_gen))
        for i, result in enumerate(results_gen):
            if print_result is True or i < print_result:
                print(result)
        if self.cache_dir is not None:
            self.logger.info('Cache saved to %s' % (saver.path))
        return events

    def process(self, jobs, retry=3, **kwargs):
        for job in progressbar(jobs):
            for retry_i in range(retry):
                try:
                    if retry_i > 0:
                        self.logger.info('Retrying job: %s (%d)', job, retry_i)
                    events = self.do_job(job, **kwargs)
                    self.events.extend(events)
                    break
                except Exception:
                    self.logger.exception('Failed job: %s', job)
                    self.reset()

    def get_output(self):
        return get_output(self.events)
