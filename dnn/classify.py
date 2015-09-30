import numpy as np
from datetime import timedelta

from sklearn.metrics import confusion_matrix
from progressbar import Counter, ProgressBar, ETA, Percentage, Bar
import time

import logging

__author__ = 'pliskowski'

logger = logging.getLogger(__name__)


def classify(net, data, do_predict, batch_size, msg='Classifying :'):
        widgets = [msg, Counter(), ' ', Bar(), ' ', Percentage(), ' ', ETA()]
        pbar = ProgressBar(widgets=widgets, maxval=round((data.size() / batch_size) + 0.5)).start()

        def pbar_decorator(func):
            def func_wrapper(*args, **kwargs):
                predictions = func(*args, **kwargs)
                pbar.update(pbar.currval + 1)
                return predictions
            return func_wrapper

        pairs = [(pbar_decorator(do_predict)(net, data_chunk), label_chunk) for data_chunk, label_chunk in data.read_batches(batch_size)]
        predictions, ground_truth = map(list, zip(*pairs))
        pbar.finish()

        y_true = np.concatenate(ground_truth)
        y_pred = np.argmax(np.vstack(predictions), axis=1)
        y_score = np.vstack(predictions)[:, 1]

        # discard padding
        pad = len(y_pred) - data.size()
        if pad:
            y_pred = y_pred[:-pad]
            y_score = y_score[:-pad]
            y_true = y_true[:-pad]

        return y_true, y_pred, y_score


def compute_acc_thresh(y_true, y_score):
    logger.debug('Started computing optimal accuracy threshold')
    s = time.time()

    y_score_s, y_true_s = zip(*np.sort(zip(y_score, y_true), axis=0))

    thresh = -np.inf
    best_acc = 0

    mat = confusion_matrix(y_true, np.ones(y_score.shape))
    tn, tp = mat[0][0], mat[1][1]

    total = len(y_true_s) * 1.0
    for i, y in enumerate(y_true_s):
        if y:
            tp -= 1
        else:
            tn += 1
        acc = (tp + tn) / total
        if acc > best_acc:
            thresh = y_score_s[i]
            best_acc = acc
    logger.debug('Finished after {} seconds'.format(timedelta(seconds=time.time() - s)))
    return thresh
