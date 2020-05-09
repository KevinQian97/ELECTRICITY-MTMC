import os.path as osp

from .log import get_logger
from .run import main, parse_args

DATA_PATH = "/mnt/hdda/kevinq/aic_20_trac3/validation"
ARGS = [osp.join(DATA_PATH,"S05"),osp.join(DATA_PATH, 'file-list.txt'), 
        osp.join(DATA_PATH, 'output.txt')]
logger = get_logger(__name__)


def test(args):
    logger.info('Testing with args: %s', args)
    return main(parse_args(args))


if __name__ == "__main__":
    test(ARGS)