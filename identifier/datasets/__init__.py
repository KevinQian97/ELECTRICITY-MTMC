from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .aic_test import aic_test
from .aic import Aic

__imgreid_factory = {
    'aic_test': aic_test,
    'Aic':Aic,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError('Invalid dataset, got "{}", but expected to be one of {}'.format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)