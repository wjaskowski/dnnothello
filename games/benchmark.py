#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import othello
import random
from subprocess import Popen, PIPE
import numpy as np
import logging

logging.basicConfig(filename="benchmark.log", filemode='w', format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S', level=logging.WARN)
log = logging.getLogger("benchmark")

class Edax:
    def __init__(self, edax_arguments):
        old_dir = os.getcwd()
        os.chdir(EDAX_BIN_PATH)
        cmd = [x for x in ('mEdax -q ' + edax_arguments).split(' ') if x != '']
        self.process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = self._read_all()
        os.chdir(old_dir)

    def reset(self, initial_board, color_to_move):
        s = othello.to_edax_str(initial_board, color_to_move) 
        self.process.stdin.write('setboard {}\n'.format(s))
        out = self._read_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.write('quit\n')

    def put_move(self, move):
        move_row, move_col = move
        assert 1 <= move_row <= othello.SIZE, "row = " + str(move_row)
        assert 1 <= move_col <= othello.SIZE, "col = " + str(move_col)

        self.process.stdin.write(b'{}{}\n'.format(chr(ord('A') + move_col - 1), move_row))
        out = self._read_all()

    def put_pass(self):
        self.process.stdin.write(b'PS\n')
        out = self._read_all()

    def _read_all(self):
        def read():
            while True:
                c = self.process.stdout.read(1)
                if c == b'>':
                    break
                yield c
        return list(read())

    def play_move(self):
        self.process.stdin.write('go\n')
        out = ''.join(self._read_all())
        #print(out)
        move_col, move_row = out[-3:-1]
        move_col = ord(move_col) - ord('A') + 1
        move_row = int(move_row)
        return move_row, move_col


def play_against_edax(edax, initial_board, color_to_move, my_color, my_agent):
    assert hasattr(my_agent, '__call__'), 'Should be my_agent(board, my_color) -> move'
    assert color_to_move in (othello.WHITE, othello.BLACK)
    assert my_color in (othello.WHITE, othello.BLACK)
    log.debug("play_againt_edax(\n" + ",".join([othello.board_to_str(initial_board), str(color_to_move), str(my_color)]))

    board = np.copy(initial_board)
    edax.reset(board, color_to_move)
    log.debug('new game')
    while True:
        log.debug('board\n' + othello.board_to_str(board))
        log.debug('color2move: ' + str(color_to_move))
        if not othello.has_valid_moves(board, color_to_move):
            color_to_move = othello.opponent(color_to_move)
            if not othello.has_valid_moves(board, color_to_move):
                break
            edax.put_pass()
            log.debug('put pass')

            continue

        if color_to_move == my_color:
            move = my_agent(board, my_color)
            edax.put_move(move)
            log.debug('my move:' + str(move))
        else:
            move = edax.play_move()
            log.debug('edax move: ' + str(move))

        assert othello.is_valid_move(board, move, color_to_move)
        othello.make_move(board, move, color_to_move)
        color_to_move = othello.opponent(color_to_move)

    score = othello.get_true_score(board)
    if my_color == othello.WHITE:
        score = reversed(score)
    return score


def benchmark_edax(edax_options, initial_states, agent):
    """ play double games """
    with Edax(edax_options) as edax:
        my_points = edax_points = wins = loses = draws = 0
        for i, (board, color_to_move) in enumerate(initial_states):
            log.debug('initial state no: ' + str(i))
            #othello.print_board(board)
            #print(color_to_move)
            for my_color in [othello.BLACK, othello.WHITE]:
                my_score, edax_score = play_against_edax(edax, board, color_to_move, my_color, agent)
                print('{:2d} {:2d}'.format(my_score, edax_score))

                my_points += my_score
                edax_points += edax_score
                if my_score > edax_score:
                    wins += 1
                elif my_score < edax_score:
                    loses += 1
                else:
                    draws += 1
        return my_points, edax_points, wins, draws, loses


random.seed(123)

def random_agent(board, my_color):
    return random.choice(list(othello.valid_moves(board, my_color)))


def read_initial_states(filename='positions.edax'):
    def to_val(x):
        if x == 'w':
            return othello.WHITE
        elif x == 'b':
            return othello.BLACK
        else:
            return othello.EMPTY

    def read_state(line):
        b = othello.new_board() 
        for i in range(1, othello.SIZE+1):
            for j in range(1, othello.SIZE+1):
                b[i,j] = to_val(line[8*(i-1)+(j-1)])
        return b, to_val(line[64])

    def read():
        with open(filename, 'r') as f:
            for line in f:
                yield read_state(line)

    return list(read())


if __name__ == '__main__':
    EDAX_BIN_PATH='/Users/Wojciech/projects/edax/edax/4.3.2/bin'

    # Edax (depth=1, opening book) vs. agent on all (1000) initial states (double games)
    my_points, edax_points, wins, draws, loses = benchmark_edax('-l 1 -book-file data/book_good.dat', read_initial_states(), random_agent)

    # Edax (depth=1, no opening book) vs. agent on all (1000) initial states (double games)
    #my_points, edax_points, wins, draws, loses = benchmark_edax('-l 1', read_initial_states(), random_agent)

    # Edax (depth=1, no opening book) vs. agent on all (1000) initial states (double games)
    #my_points, edax_points, wins, draws, loses = benchmark_edax('-l 21 -book-file data/book_good.dat', read_initial_states(), random_agent)

    print('\nFinal result:')
    print(my_points, edax_points, wins, draws, loses)
