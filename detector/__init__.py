from functools import partial

from .detectron2 import MaskRCNN as Detector
from .stage import DetectorStage as DetectorStage_base

__all__ = ['Detector', 'DetectorStage']

DetectorStage = partial(DetectorStage_base, Detector)
