from os.path import join
import numpy as np
from sklearn.externals import joblib
from dnn.nets import load_net, predict
from games import othello
from dnn.dataset import encode_channels

__author__ = 'pliskowski'


def create_dnn_agent(model_dir, model_name, input_size=8, channels=2, batch_size=1, model_iter=30000, gpu=1):
    net = load_net(join(model_dir, model_name), model_name, input_size, batch_size, channels, model_iter, gpu)
    # net has learned transformed labels, so get the encoder
    encoder = joblib.load(join(model_dir, model_name, 'dataset', 'encoder.pkl'))

    def get_move(probabilities):
        prediction = np.argmax(probabilities)
        probabilities[prediction] = 0
        move = encoder.inverse_transform(prediction)
        move = othello.decode_move(move)
        return move

    def dnn_agent(board, my_color):
        if my_color == othello.WHITE:
            board = othello.inverted(board)
            my_color = othello.opponent(my_color)
        probabilities = predict(net, [encode_channels(board[1:-1, 1:-1], my_color)])[0]
        move = get_move(probabilities)
        while not othello.is_valid_move(board, move, my_color):
            move = get_move(probabilities)
        return move

    return dnn_agent
