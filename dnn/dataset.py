import gzip
from itertools import islice
import numpy as np
import os
import cPickle as pickle
import logging

from sklearn.externals import joblib
from os.path import join

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from games import othello
from games.othello import add_walls
from games.othello_data import generate_learning_data

from util.io import LMDBReader, save_lmdb
from dnn.nets import deploy_model
from util.utils import create_dir

__author__ = 'pliskowski'

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] '
                           '(%(filename)s:%(lineno)d) -- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)


def split_train_test(experiment_dir, data_path, model_type, solver_type, train_batch, test_batch, train_iters,
                     test_interval, lr, gamma, step, test_size=0.33, seed=123):
    model_dir = join(experiment_dir, model_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    dataset_dir = join(model_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    reader = LMDBReader(data_path)
    data, labels = reader.read_all()

    encoder = get_label_encoder(labels)
    labels = encoder.transform(labels)
    joblib.dump(encoder, join(dataset_dir, 'encoder.pkl'))

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=seed)

    train_set_size = len(y_train)
    test_set_size = len(y_test)

    logger.info('Creating lmdbs: train_size={} test_size={}'.format(train_set_size, test_set_size))
    save_lmdb(join(dataset_dir, 'train'), x_train, y_train)
    save_lmdb(join(dataset_dir, 'test'), x_test, y_test)

    deploy_model(model_dir, dataset_dir, model_type, solver_type, train_batch, test_batch, test_set_size, test_interval,
                 train_iters, model_type, lr, gamma, step)


def get_num_unique_labels():
    reader = LMDBReader('../games/train')
    data, labels = reader.read_all()

    print np.unique(labels)
    print len(np.unique(labels))


def get_label_encoder(labels=None):
    if labels is None:
        reader = LMDBReader('../games/train')
        _, labels = reader.read_all()
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    return le


def encode_channels(board, player):
    """
    Encodes the board into 2x8x8 binary matrix.
    The first matrix has ones indicating the fields occupied by the player who is about to play.
    The second matrix has ones where the opponent's pieces are.

    Returns 2x8x8 array.
    """
    c1 = board == player
    c2 = board == othello.opponent(player)
    return np.concatenate((c1[None, ...], c2[None, ...]), axis=0)


def encode_valid_moves(board, player):
    """
    Encodes the board into 3x8x8 binary matrix.
    The third channels contains ones for every valid moves.
    """

    def mark_valid_moves(board):
        moves = np.zeros(board.shape, dtype=np.uint8)
        for r, c in othello.valid_moves(board, player):
            moves[r, c] = 1
        return moves[1:-1, 1:-1]

    c1 = board == player
    c2 = board == othello.opponent(player)
    c3 = mark_valid_moves(add_walls(board))
    return np.concatenate((c1[None, ...].astype(np.uint8), c2[None, ...].astype(np.uint8), c3[None, ...]), axis=0)


def create_dataset(out, data_path, encoder=encode_channels, shape=(2, 8, 8), lmdb=False):
    create_dir(out)
    data = pickle.load(gzip.open(data_path, 'rb'))

    x = np.zeros(((len(data),) + shape), dtype=np.uint8)
    y = np.zeros(len(data), dtype=np.uint8)
    for i, (board, player, move) in enumerate(data):
        x[i] = encoder(board, player)
        y[i] = move
    if lmdb:
        save_lmdb(out, x, y)
    return x, y


if __name__ == '__main__':
    data_path = '/home/pliskowski/Documents/repositories/dlothello/games/data_sym.dump'

    logger.info('Started building LMDB for sym')
    create_dataset('./new-train-sym', data_path, encode_channels, shape=(2, 8, 8), lmdb=True)

    logger.info('Started building LMDB for sym-vmoves')
    create_dataset('./new-train-sym-vmoves', data_path, encode_valid_moves, shape=(3, 8, 8), lmdb=True)

    logger.info('Done')

    # data_path = '/home/pliskowski/Documents/repositories/dlothello/games/data.dump'
    # logger.info('Started building LMDB for original')
    # create_dataset('./original', data_path, encode_channels, shape=(2, 8, 8), lmdb=True)