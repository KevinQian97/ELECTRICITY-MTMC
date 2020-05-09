from functools import partial

from .stage import MonitorStage as MonitorStage_base
from .movement import MovementMonitor as Monitor

__all__ = ['Monitor', 'MinotorStage']

MinotorStage = partial(MonitorStage_base, Monitor)
