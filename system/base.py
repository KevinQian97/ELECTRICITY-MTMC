import resource
from typing import List, Union

from ..pipeline import LoggedTask, Pipeline, Stage
from ..utils import get_logger

resource.setrlimit(resource.RLIMIT_NOFILE, (1048576, 1048576))


class System(object):

    def __init__(self, stages: List[Stage], video_dir: str,
                 cache_dir: Union[None, str] = None, batch_size: int = 1,
                 stride: int = 1):
        self.stages = stages
        self.video_dir = video_dir
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.stride = stride
        self.logger = get_logger(self.__class__.__name__)
        self.pipeline = Pipeline(self.stages)

    def init(self, task: LoggedTask, **kwargs):
        self.logger.info('System initializing ...')
        self.pipeline.init(task, **kwargs)
        self.logger.info('System initialized.')

    def process(self, task: LoggedTask, **kwargs):
        self.logger.info('Running task: %s' % (repr(task.value)))
        yield from self.pipeline.process(task, **kwargs)
        self.logger.info('Finished task: %s' % (repr(task.value)))

    def finish(self):
        self.pipeline.close()

    def reset(self):
        self.logger.info('System resetting')
        self.pipeline.close()
        self.pipeline = Pipeline(self.stages)
        self.logger.info('System resetted')

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.video_dir)
