from functools import partial

from .deepsort import DeepSort as Tracker
from .stage import TrackerStage as TrackerStage_base

__all__ = ['Tracker', 'TrackerStage']

TrackerStage = partial(TrackerStage_base, Tracker)
  
