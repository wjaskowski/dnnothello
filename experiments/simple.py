import argparse
from os.path import join
from dnn.dataset import split_train_test
from dnn.train import train_and_test

__author__ = 'pliskowski'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Othello move prediction experiment')
    parser.add_argument('--out_path', required=True)
    parser.add_argument('--data_path', required=True)

    parser.add_argument('--model_type', default='cnn_nopool')
    parser.add_argument('--solver_type', default='sgd')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--opt_acc', action='store_true')
    parser.add_argument('--train_batch', type=int, default=256)
    parser.add_argument('--test_batch', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=1000)
    parser.add_argument('--test_percent', type=float, default=0.2)

    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--iters', type=int, default=30000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=0.0005)
    parser.add_argument('--step', type=int, default=10000)
    parser.add_argument('--steps', type=int, nargs='+', default=[10000, 20000, 30000])

    args = parser.parse_args()

    # create train and test sets
    split_train_test(args.out_path, args.data_path, args.model_type, args.solver_type, args.train_batch,
                     args.test_batch, args.iters, args.test_interval, args.test_percent, args.lr, args.gamma, args.step,
                     args.steps, args.decay)

    # train the networks
    if args.train:
        train_and_test(args.out_path, join(args.out_path, 'out'), args.gpu,
                       args.test_batch, opt_acc=args.opt_acc)
