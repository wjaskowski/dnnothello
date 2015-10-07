#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np

WALL = -1
WHITE = 0
EMPTY = 1
BLACK = 2

SIZE = 8
WIDTH = SIZE + 2

DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def new_board():
    board = np.ones((WIDTH, WIDTH), np.int8) * EMPTY
    M = WIDTH // 2
    board[M - 1, M - 1] = WHITE
    board[M - 1, M] = BLACK
    board[M, M] = WHITE
    board[M, M - 1] = BLACK

    board[[0, WIDTH - 1], :] = WALL
    board[:, [0, WIDTH - 1]] = WALL
    return board


def make_move(board, move, player):
    move_row, move_col = move
    assert 1 <= move_row <= SIZE, "row = " + str(move_row)
    assert 1 <= move_col <= SIZE, "col = " + str(move_col)
    assert player in (WHITE, BLACK)

    for dirx, diry in DIRS:
        x = move_col + dirx
        y = move_row + diry

        while board[y, x] == opponent(player):
            x += dirx
            y += diry

        if board[y, x] == player:
            x -= dirx
            y -= diry
            while board[y, x] == opponent(player):
                board[y, x] = player
                x -= dirx
                y -= diry
    board[move_row, move_col] = player


def is_valid_move(board, move, player):
    move_row, move_col = move
    assert 1 <= move_row <= SIZE
    assert 1 <= move_col <= SIZE
    assert player in (WHITE, BLACK)

    if board[move_row, move_col] != EMPTY:
        return False

    for dirx, diry in DIRS:
        x = move_col + dirx
        y = move_row + diry

        while board[y, x] == opponent(player):
            x += dirx
            y += diry

        if board[y, x] == player and board[y - diry, x - dirx] == opponent(player):
            return True
    return False


def valid_moves(board, player):
    for row in range(1, SIZE + 1):
        for col in range(1, SIZE + 1):
            if is_valid_move(board, (row, col), player):
                yield row, col


def has_valid_moves(board, player):
    return len(list(valid_moves(board, player))) > 0


def encode_move(row, col):
    return row * WIDTH + col


def decode_move(move):
    return move // WIDTH, move % WIDTH


def opponent(player):
    assert player in (WHITE, BLACK)
    return WHITE if player == BLACK else BLACK


def get_score(board):
    b = (board[1:-1, 1:-1] == BLACK).sum()
    w = (board[1:-1, 1:-1] == WHITE).sum()
    return b, w


def get_true_score(board):
    b, w = get_score(board)
    e = SIZE * SIZE - b - w
    if b > w:
        return b + e, w
    elif b < w:
        return b, w + e
    else:
        return b + e // 2, w + e // 2


def print_board(board):
    for row in range(1,SIZE+1):
        for col in range(1,SIZE+1):
            if board[row][col] == BLACK:
                print('* ',end='')
            elif board[row][col] == WHITE:
                print('0 ',end='')
            else:
                print('- ',end='')
        print()
