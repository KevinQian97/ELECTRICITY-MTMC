import heapq
import time
from collections import namedtuple

from ..detector.base import Frame
from ..pipeline import LoggedTask, Stage

MonitorResult = namedtuple('MonitorResult', ['frame', 'events'])


class MonitorStage(Stage):

    def __init__(self, monitor_class, forward_frame=False):
        super().__init__()
        self.monitor_class = monitor_class
        self.forward_frame = forward_frame
        self.monitor = None

    def pipeline_fn(self, res, task):
        super().pipeline_fn(res, task)
        try:
            assert isinstance(task, LoggedTask)
            if self.monitor is None:
                self.monitor = self.monitor_class(
                    task.meta['video_name'], task.meta['fps'],
                    task.meta['stride'], task.meta['video_id'],
                    task.meta['camera_id'], task.meta['image_size'][0],
                    task.meta['image_size'][1])
                self.logger.debug('Started monitor: %s', self.monitor)
            return_task = LoggedTask(prev_task=task)
            if task.meta.get('end', False):
                events = self.monitor.finish()
                self.logger.debug('Finished monitor: %s', self.monitor)
                self.monitor = None
                result = MonitorResult(None, events)
                return_task.finish(result, self.monitor)
                return return_task
            assert isinstance(task.value, Frame)
            frame = task.value
            events = self.monitor.monit(frame)
            if not self.forward_frame:
                frame = None
            result = MonitorResult(frame, events)
            return_task.finish(result, self.monitor)
            return return_task
        except Exception as e:
            if self.monitor is None:
                raise e
            self.logger.exception('Task failed: %s', task)
