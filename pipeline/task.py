import os
import time
from collections import namedtuple

from .easy_pipeline import ValuedTask

PROFILING = 'profiling' in os.environ

ProcessorItem = namedtuple('ProcessorItem', ['processor', 'duration'])


class LoggedTask(ValuedTask):

    def __init__(self, value=None, prev_task=None, meta=None,
                 start_time=None, processor=None, profiling=PROFILING):
        assert not (prev_task is None and meta is None)
        super().__init__(value)
        self.profiling = profiling
        self.finished = False
        if meta is None:
            meta = prev_task.meta
        self.meta = meta
        if self.profiling:
            self.meta = meta.copy()
            self.meta['processors'] = meta.get('processors', []).copy()
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
            self.finish(value, processor)

    def finish(self, value, processor=None):
        assert not self.finished
        self.finished = True
        self.value = value
        if self.profiling:
            duration = time.time() - self.start_time
            processor = repr(processor)
            processor_item = ProcessorItem(processor, duration)
            self.meta['processors'].append(processor_item)

    def __repr__(self):
        string = ''
        if self.meta is not None:
            string = repr(self.meta)
        return 'LoggedTask(%s)' % (string)
