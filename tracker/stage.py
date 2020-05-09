from collections import deque
import time

from ..detector.base import Frame
from ..pipeline import LoggedTask, Stage


class TrackerStage(Stage):

    def __init__(self, tracker_class):
        super().__init__()
        self.tracker_class = tracker_class
        self.tracker = None
        self.end_task = None

    def init(self, task):
        self.tracker = self.tracker_class(
            task.meta['video_name'], task.meta['fps'])
        self.end_task = None
        self.reorder_buffer = deque()
        self.next_frame_id = task.meta['start_id']
        self.last_frame_id = None

    def push(self, task):
        offset = task.meta['image_id'] - self.next_frame_id
        if offset >= len(self.reorder_buffer):
            self.reorder_buffer.extend([None] * (
                offset - len(self.reorder_buffer) + 1))
        self.reorder_buffer[offset] = task

    def iter_pop(self):
        while len(self.reorder_buffer) > 0:
            if self.reorder_buffer[0] is not None:
                task = self.reorder_buffer.popleft()
                assert task.meta['image_id'] == self.next_frame_id
                yield_task = LoggedTask(prev_task=task)
                self.next_frame_id = task.meta['image_id'] + 1
                frame = task.value
                frame = self.tracker.track(frame)
                yield_task.finish(frame, self.tracker)
                yield yield_task
            else:
                return

    def finish_at(self, end_task):
        self.end_task = end_task
        self.last_frame_id = end_task.meta['last_image_id']

    def finished(self):
        if self.last_frame_id is None:
            return False
        return self.next_frame_id > self.last_frame_id

    def pipeline_fn(self, res, task):
        super().pipeline_fn(res, task)
        try:
            assert isinstance(task, LoggedTask)
            if self.tracker is None:
                self.init(task)
                self.logger.debug('Started tracker: %s', self.tracker)
            if task.meta.get('end', False):
                self.finish_at(task)
            else:
                assert isinstance(task.value, Frame)
                self.push(task)
                if len(self.reorder_buffer) > 100:
                    self.logger.warn(
                        'Reorder buffer size: %d', len(self.reorder_buffer))
            yield from self.iter_pop()
            if self.finished():
                yield LoggedTask(
                    None, prev_task=self.end_task, start_time=time.time(),
                    processor=self.tracker)
                self.logger.debug('Finished tracker: %s', self.tracker)
                self.tracker = None
                assert len(self.reorder_buffer) == 0
        except Exception as e:
            if self.tracker is None:
                raise e
            self.logger.exception('Task failed: %s', task)
