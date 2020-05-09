from __future__ import absolute_import

import sys
import os
import os.path as osp
from .io import mkdir_if_missing


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class RankLogger(object):
    """
    RankLogger records the rank1 matching accuracy obtained for each
    test dataset at specified evaluation steps and provides a function
    to show the summarized results, which are convenient for analysis.
    Args:
    - source_names (list): list of strings (names) of source datasets.
    - target_names (string): name of the target datasets.
    """
    def __init__(self, source_names, target_name):
        self.source_names = source_names
        self.target_name = target_name
        self.logger = {'epoch': [], 'rank1': []}

    def write(self, name, epoch, rank1):
        self.logger['epoch'].append(epoch)
        self.logger['rank1'].append(rank1)

