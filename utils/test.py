import os.path as osp
import os
from .log import get_logger
from .run import main, parse_args


cur_path = os.getcwd()
DATA_PATH = "./ELECTRICITY-MTMC/datasets/aic_20_trac3/test"
OUT_PATH = os.path.join(cur_path,"ELECTRICITY-MTMC","exp")
if not osp.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

if not osp.exists(DATA_PATH):
    raise RuntimeError("cant find the aicity dataset")
ARGS = [osp.join(DATA_PATH,"S06"),osp.join(OUT_PATH, 'file-list.txt'), 
        osp.join(OUT_PATH, 'tracklets.txt')]
logger = get_logger(__name__)


def test(args):
    logger.info('Testing with args: %s', args)
    return main(parse_args(args))


if __name__ == "__main__":
    test(ARGS)
