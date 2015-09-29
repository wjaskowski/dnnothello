#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

import struct
import glob
import os
import pickle
import gzip

import numpy as np
import sys

import othello


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
        yield from read_wthor(wtb)


def generate_learning_data(path):
    games = read_all(path)
    for i, (true_score, moves) in enumerate(games):
        if i % 1000 == 0: print(i)
        yield from moves2data(true_score, moves)


def moves2data(true_score, moves):
    board = othello.new_board()
    player = othello.BLACK
    moves = [m for m in moves if m != 0]  # remove (trailing) zeros
    for move in moves:
        row, col = move // othello.WIDTH, move % othello.WIDTH
        if not othello.is_valid_move(board, (row, col), player):
            player = othello.opponent(player)
            assert othello.is_valid_move(board, (row, col), player)
        yield np.copy(board[1:-1, 1:-1]), player, move
        othello.make_move(board, (row, col), player)
        player = othello.opponent(player)
    if othello.get_true_score(board)[0] != true_score:
        print("WARNING: {} != {}".format(othello.get_true_score(board), true_score))


def encode_channels(board, player):
    """
    Encodes the board in two 8x8 binary matrices.
    The first matrix has ones indicating the fields occupied by the player who is about to play.
    The second matrix has ones where the opponent's pieces are.

    Returns 2x8x8 array.
    """
    c1 = board == player
    c2 = board == othello.opponent(player)
    return np.concatenate((c1[None, ...], c2[None, ...]), axis=0).astype(np.uint8)


def get_learning_data():
    return pickle.load(gzip.open('data.dump', 'rb'))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        data = list(generate_learning_data('wthor_data'))
        pickle.dump(data, gzip.open('data.dump', 'wb'))

    data = get_learning_data()
    print("read:", len(data), "positions")
    print("example:", data[0])
