#!/usr/bin/env python
# -*- coding: utf-8 -*-
import struct
import glob
import os
import pickle
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
        if i>0 and i % 1000 == 0:
            break


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


def encode_channels(board, player):
    """
    Encodes the board in two 8x8 binary matrices.
    The first matrix has ones indicating the fields occupied by the player who is about to play.
    The second matrix has ones where the opponent's pieces are.

    Returns 2x8x8 array.
    """
    c1 = board == player
    c2 = board == othello.opponent(player)
    return np.concatenate((c1[None, ...], c2[None, ...]), axis=0)


def create_dataset(out, lmdb=False):
    data = list(generate_learning_data('wthor_data'))

    x = np.zeros(((len(data),) + (2, 8, 8)), dtype=np.uint8)
    y = np.zeros(len(data), dtype=np.uint8)
    for i, (board, player, move) in enumerate(data):
        x[i] = encode_channels(board, player)
        y[i] = move
    if lmdb:
        from util.io import save_lmdb
        save_lmdb(out, x, y)
    return x, y


def get_learning_data():
    return pickle.load(open('data2.dump', 'rb'))


def data_from_black_perspective(data):
    for board,player,move in data:
        if player == othello.BLACK: 
            yield board, player, move
        else:
            yield othello.inverted(board), othello.BLACK, move


def removed_duplicates(data):
    x = sorted([(list(x.flatten()),y,z) for x,y,z in data])
    for i in range(1, len(x)):
        if x[i-1][0:2] != x[i][0:2] or x[i-1][2] != x[i][2]:
            yield data[i]


def main_remove_duplicates(data): 
    print("input: {} positions".format(len(data)))
    data = list(data_from_black_perspective(data))
    data = list(removed_duplicates(data))
    random.shuffle(data)
    print("output: {} positions".format(len(data)))
    pickle.dump(data, gzip.open('data_nodup.dump', 'wb'))
    print("finished")

def data_with_symmetries(data):
    for board,player,move in data:
        for sym_board, sym_move in zip(othello.symmetric(board),othello.symmetric_move(board)):
            yield sym_board,player,sym_move

if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        data = list(generate_learning_data('wthor_data'))
        pickle.dump(data, gzip.open('data.dump', 'wb'))

    main_remove_duplicates(pickle.load(open('data.dump', 'rb')))

    #data = pickle.load(gzip.open('data_nodup.dump', 'rb'))

    #data = list(remove_duplicates(data_with_symmetries(data[:1000])))
    #print(len(data))

    #pickle.dump(data, gzip.open('data_nodup_sym.dump', 'wb'))

    #print('end')
