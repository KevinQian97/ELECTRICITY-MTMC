from ..utils import get_logger
from .easy_pipeline import PipelineItem
from .task import LoggedTask


class Stage(object):

    def __init__(self, name=''):
        self.name = name
        self.logger = None

    def pipeline_fn(self, res, task: LoggedTask) -> LoggedTask:
        if self.logger is None:
            self.logger = get_logger(self.__class__.__name__ + self.name)

    def pipeline_init_fn(self):
        return

    def get_worker_num(self) -> int:
        return 1

    def get_queue_size(self) -> int:
        return 16

    def get_pipeline_item(self):
        pipeline_item = PipelineItem(
            self.pipeline_fn, self.pipeline_init_fn,
            self.get_worker_num(), self.get_queue_size(), 
            self.__class__.__name__)
        return pipeline_item
