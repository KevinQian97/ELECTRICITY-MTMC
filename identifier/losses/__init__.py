from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .xeloss import CrossEntropyLoss
from .tripletloss import TripletLoss


def DeepSupervision(criterion, xs, y):
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss