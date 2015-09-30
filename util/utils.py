import os
import re
import numpy as np


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def chunks(iterable, chunk_size):
    for i in xrange(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]


def padded_chunks(iterable, batch_size):
    num = len(iterable)
    remainder = num % batch_size
    num_batches = num / batch_size

    # yield full batches
    for b in range(num_batches):
        i = b * batch_size
        yield iterable[i:i + batch_size]

    # yield last padded batch, if any
    if remainder > 0:
        item_shape = iterable[0].shape
        padding = np.zeros((batch_size - remainder,) + item_shape)
        yield np.concatenate([iterable[-remainder:], padding])
