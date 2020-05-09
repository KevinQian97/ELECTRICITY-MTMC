import time
import os.path as osp
from collections import namedtuple

from ..pipeline import LoggedTask, Stage

LoaderTask = namedtuple('LoaderTask', [
    'video_name', 'video_dir', 'video_id', 'camera_id', 'batch_size', 'limit', 'stride', 'start_id'], defaults=[1, None, 1, 0])


class LoaderStage(Stage):

    def __init__(self, loader_classes):
        super().__init__()
        self.loader_classes = loader_classes

    def pipeline_fn(self, res, task):
        super().pipeline_fn(res, task)
        try:
            last_task = None
            assert isinstance(task, LoggedTask)
            assert isinstance(task.value, LoaderTask)
            loader_task = task.value
            video_format = osp.splitext(loader_task.video_name)[1][1:]
            loader_class = self.loader_classes[video_format]
            loader = loader_class(
                loader_task.video_name, loader_task.video_dir)
            self.logger.debug('Started loader: %s', loader)
            meta = {**loader_task._asdict(),
                    'fps': loader.fps / loader_task.stride}
            if loader_task.start_id != 0:
                loader.seek(loader_task.start_id)
            yield_task = LoggedTask(prev_task=task, meta=meta)
            for images, image_ids in loader.read_iter(
                    loader_task.batch_size, loader_task.limit,
                    loader_task.stride):
                yield_task.meta['image_ids'] = image_ids
                yield_task.meta['image_size'] = images[0].shape
                yield_task.finish(images, loader)
                yield yield_task
                last_task = yield_task
                yield_task = LoggedTask(prev_task=task, meta=meta)
        except GeneratorExit:
            self.logger.warn(
                'Task stopped before complete: %s', loader_task)
        except Exception:
            self.logger.exception('Task failed: %s', loader_task)
        finally:
            if last_task is not None:
                yield_task = LoggedTask(
                    None, prev_task=task, meta=meta,
                    start_time=time.time(), processor=loader)
                yield_task.meta['end'] = True
                yield_task.meta['last_image_id'] = last_task.meta[
                    'image_ids'][-1]
                yield yield_task
            self.logger.debug('Finished loader: %s', loader)
