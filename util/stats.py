# coding=utf-8

import math
import numpy as np
from scipy import stats


__author__ = 'pliskowski'


def ci(seq):
    n, min_max, mean, var, skew, kurt = stats.describe(seq)
    std = math.sqrt(var)
    r = stats.t.interval(0.95, len(seq) - 1, loc=mean, scale=std / math.sqrt(len(seq)))
    return '{} ± {}'.format(round(mean, 4), round((r[1] - r[0]) / 2, 4))


def se(x):
    return x.std() / np.sqrt(len(x))

