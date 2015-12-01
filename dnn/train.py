import argparse
import time
import os
from datetime import timedelta
from os.path import join

import logging
from dnn.classify import compute_acc_thresh, classify
from dnn.evaluate import ReportBuilder
from dnn.nets import get_solver, predict
from util.dbio import LMDBReader
from util.redirector import Redirector
from util.utils import natural_sort_key

logger = logging.getLogger(__name__)

__author__ = 'pliskowski'


def train_and_test(experiment_dir, out_dir, gpu, batch_size, models=None, opt_acc=False):
    """
    Trains the networks found in the experiment directory, sequentially one after another. Only one gpu is used,
    and evaluation is performed on the basis of test set found in the data directory.
    """
    if models is None:
        models = sorted(next(os.walk(experiment_dir))[1], key=natural_sort_key)

    report = ReportBuilder(experiment_dir)
    for model_name in models:
        y, pred, score, thr = train_and_test_one(experiment_dir, model_name, batch_size, gpu, opt_acc)
        report.add(model_name, y, pred, score, thr)
    report.generate(out_dir)


def train_and_test_one(experiment_dir, model_name, batch_size=100, gpu=0, opt_acc=False):
    """
    Trains and then evaluates a single net. Returns the evaluation result.
    """
    model_dir = join(experiment_dir, model_name)
    dataset_dir = join(model_dir, 'dataset')

    # train on the data
    logger.info('Started training model: {}'.format(model_name))
    with Redirector(stderr=join(model_dir, 'log.txt'), mode='w'):
        solver = get_solver(model_dir, gpu)
        s = time.time()
        solver.solve()
    logger.info('Training took: {}'.format(timedelta(seconds=time.time() - s)))

    # get the test net
    net = solver.test_nets[0]

    # evaluate on the training set
    train_thresh = None
    if opt_acc:
        trainset = LMDBReader(join(dataset_dir, 'train'))
        y_true, y_pred, y_score = classify(net, trainset, predict, batch_size, msg='Classifying train instances: ')
        train_thresh = compute_acc_thresh(y_true, y_score)

    # evaluate the net on the testset
    logger.info('Evaluating model: {} on the testset'.format(model_name))
    s = time.time()
    testset = LMDBReader(join(dataset_dir, 'test'))
    y_true, y_pred, y_score = classify(net, testset, predict, batch_size, msg='Classifying test instances: ')
    logger.info('Evaluation took: {}'.format(timedelta(seconds=time.time() - s)))

    return y_true, y_pred, y_score, train_thresh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and test the networks')
    parser.add_argument('--exp_path', required=True)
    parser.add_argument('--out_dir', default='out')

    parser.add_argument('--models', nargs='+')
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--opt_acc', action='store_true')

    parser.add_argument('--gpu', type=int, default=1)

    args = parser.parse_args()
    train_and_test(args.exp_path, join(args.out_dir, 'out'), args.gpu, args.test_batch,
                   models=args.models, opt_acc=args.opt_acc)
