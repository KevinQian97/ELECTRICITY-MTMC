import os.path as osp
from collections import namedtuple
from typing import Iterator, List, Tuple, Union

import torch


class Loader(object):

    def __init__(self, video_path: str, parent_dir: str = ''):
        self.path = osp.join(parent_dir, video_path)

    def read_iter(self, batch_size: int = 1, limit: Union[None, int] = None,
                  stride: int = 1,  start: int = 0) \
            -> Iterator[Tuple[List[torch.Tensor], List[int]]]:
        ...

    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.path)
