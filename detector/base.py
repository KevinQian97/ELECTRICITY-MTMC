import time
from collections import namedtuple
from enum import IntEnum, auto
from typing import List, Union

import torch

Frame = namedtuple('Frame', ['image_id', 'image', 'instances'])


class ObjectType(IntEnum):
    Car = auto()
    Truck = auto()
    Person = auto()
    Bike = auto()


class Detector(object):

    def __init__(self, gpu_id: Union[None, int] = None):
        self.device = 'cpu'
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = 'cuda:%d' % (gpu_id)

    def detect(self, images: List[torch.Tensor],
               image_ids: List[int]) -> List[Frame]:
        raise NotImplementedError
        
    def __repr__(self):
        return '%s.%s@%s' % (
            self.__module__, self.__class__.__name__, self.device)
