#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
import othello
import random
from subprocess import Popen, PIPE


class Edax:
    def __init__(self, edax_arguments):
        old_dir = os.getcwd()
        os.chdir(EDAX_BIN_PATH)
        cmd = [x for x in ('mEdax -q ' + edax_arguments).split(' ') if x != '']
        self.process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out = self._read_all()
        os.chdir(old_dir)

    def reset(self, initial_board, player_to_move):
        s = othello.to_edax_str(initial_board, player_to_move) 
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
        move_col, move_row = out[-3:-1]
        move_col = ord(move_col) - ord('A') + 1
        move_row = int(move_row)
        return move_row, move_col


def play_against_edax(edax, initial_board, player_to_move, my_color, my_agent):
    assert hasattr(my_agent, '__call__'), 'Should be my_agent(board, my_color) -> move'
    assert player_to_move in (othello.WHITE, othello.BLACK)
    assert my_color in (othello.WHITE, othello.BLACK)

    board = initial_board 
    edax.reset(board, player_to_move)
    while True:
        #othello.print_board(board)
        if not othello.has_valid_moves(board, player_to_move):
            player_to_move = othello.opponent(player_to_move)
            edax.put_pass()

            if not othello.has_valid_moves(board, player_to_move):
                break

        if player_to_move == my_color:
            move = my_agent(board, my_color)
            edax.put_move(move)
        #    print('my move', move)
        else:
            move = edax.play_move()
        #    print('edax move', move)

        assert othello.is_valid_move(board, move, player_to_move)
        othello.make_move(board, move, player_to_move)
        player_to_move = othello.opponent(player_to_move)

    score = othello.get_score(board)
    if my_color == othello.WHITE:
        score = score[::-1]
    return score


def benchmark_edax(edax_options, num_games, agent):
    with Edax(edax_options) as edax:
        my_points = edax_points = wins = loses = draws = 0
        for i in range(num_games):
            my_color = othello.BLACK if i%2==0 else othello.WHITE
            res = play_against_edax(edax, othello.new_board(), othello.BLACK, my_color, agent)
            print('{:2d} {:2d}'.format(*res))

            my_points += res[0]
            edax_points += res[1]
            if res[0] > res[1]:
                wins += 1
            elif res[0] < res[1]:
                loses += 1
            else:
                draws += 1
        return my_points, edax_points, wins, draws, loses


random.seed(123)

def random_agent(board, my_color):
    return random.choice(list(othello.valid_moves(board, my_color)))


if __name__ == '__main__':
    EDAX_BIN_PATH='/Users/Wojciech/projects/edax/edax/4.3.2/bin'
    #my_points, edax_points, wins, draws, loses = benchmark_edax('-l 1' , 10, random_agent) # Depth=1
    my_points, edax_points, wins, draws, loses = benchmark_edax('-l 1 -book-file data/book_good.dat', 10, random_agent) # Depth=1
    #my_points, edax_points, wins, draws, loses = benchmark_edax('-l 21', 10, random_agent)

    print('\nFinal result:')
    print(my_points, edax_points, wins, draws, loses)
