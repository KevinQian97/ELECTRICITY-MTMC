import logging
import os
from functools import partial

from tqdm.autonotebook import tqdm

progressbar = partial(tqdm, dynamic_ncols=True)

DEFAULT_LEVEL = logging.INFO if not 'profiling' in os.environ else logging.DEBUG


def get_logger(name, level=DEFAULT_LEVEL, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if len(logger.handlers) > 0:
        return logger
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s')
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
