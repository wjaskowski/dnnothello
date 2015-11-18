#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import logging
import struct
import glob
import os
import cPickle as pickle
import gzip
import numpy as np
import sys
import random

from games import othello


def read_wthor(file_name):
    with open(file_name, mode='rb') as f:
        header = struct.unpack("<BBBBIHHBBBB", f.read(16))
        games = header[4]
        year = header[6]
        board_size = header[7]
        assert board_size == 0 or board_size == 8

        for i in range(games):
            game_info = struct.unpack("<HHHBB", f.read(8))
            true_score = game_info[3]
            moves = struct.unpack('<' + 'B' * 60, f.read(60))
            yield true_score, moves


def read_all(path):
    for wtb in glob.glob(os.path.join(path, '*.wtb')):
        for true_score, moves in read_wthor(wtb):
            yield true_score, moves


def generate_learning_data(path='wthor_data'):
    games = read_all(path)
    for i, (true_score, moves) in enumerate(games):
        if i % 1000 == 0: print(i)
        for board, player, move in moves2data(true_score, moves):
            yield board, player, move
        #if i % 1000 == 0:
        #    break


def moves2data(true_score, moves):
    board = othello.new_board()
    player = othello.BLACK
    for i in range(len(moves)-1):
        if moves[i] == 0:
            assert moves[i+1] == 0

    moves = [m for m in moves if m != 0]  # remove (trailing) zeros
    for move in moves:
        row, col = othello.decode_move(move)
        if not othello.is_valid_move(board, (row, col), player):
            player = othello.opponent(player)
            assert othello.is_valid_move(board, (row, col), player)
        yield np.copy(board[1:-1, 1:-1]), player, move
        othello.make_move(board, (row, col), player)
        player = othello.opponent(player)
    if othello.get_true_score(board)[0] != true_score:
        print("TRUE SCORE WARNING: {} != {}".format(othello.get_true_score(board)[0], true_score))


def get_learning_data():
    return pickle.load(open('data2.dump', 'rb'))


def data_from_black_perspective(data):
    for board,player,move in data:
        if player == othello.BLACK: 
            yield board, player, move
        else:
            yield othello.inverted(board), othello.BLACK, move


def removed_duplicates(data):
    N = othello.SIZE
    def coding_board():
        a = 1
        coding = np.zeros(N*N//2)
        for i in range(N*N//2):
            coding[i] = a
            a *= 3
        return coding 

    coding = coding_board()

    def compress(board):
        x = board.flatten()
        return sum(x[:N*N//2] * coding), sum(x[N*N//2:] * coding)

    x = [(compress(x), y, z, x) for x, y, z in data]
    x.sort(key=lambda x: x[:3])
    yield x[0][3], x[0][1], x[0][2]
    for i in range(1, len(x)):
        if x[i-1][0:2] != x[i][0:2] or x[i-1][2] != x[i][2]:
            yield x[i][3], x[i][1], x[i][2]


def data_with_symmetries(data):
    for board, player, raw_move in data:
        move = othello.decode_move(raw_move)
        for sym_board, sym_move in zip(othello.symmetric(board), othello.symmetric_move(move)):
            yield sym_board, player, othello.encode_move(*sym_move)


# First experiment 
def main_remove_duplicates(): 
    data = pickle.load(gzip.open('data.dump', 'rb'))
    logger.info("input: {} positions".format(len(data)))
    data = list(data_from_black_perspective(data))
    data = list(removed_duplicates(data))
    random.shuffle(data)
    logger.info("output: {} positions".format(len(data)))
    pickle.dump(data, gzip.open('data_nodup.dump', 'wb'))
    logger.info("finished")


# Second experiment 
def main_extended_symmetric(): 
    data = pickle.load(gzip.open('data_nodup.dump', 'rb'))
    logger.info("input: {} positions".format(len(data)))
    data = list(data_with_symmetries(data))
    data = list(removed_duplicates(data))
    random.shuffle(data)
    logger.info("output: {} positions".format(len(data)))
    pickle.dump(data, gzip.open('data_sym.dump', 'wb'))
    logger.info("finished")


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(threadName)s] '
                           '(%(filename)s:%(lineno)d) -- %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
random.seed(123)

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        data = list(generate_learning_data('wthor_data'))
        pickle.dump(data, gzip.open('data.dump', 'wb'))

    main_extended_symmetric()
    #main_remove_duplicates()
