from os.path import join
import numpy as np
from sklearn.externals import joblib
from dnn.nets import load_net, predict
from games import othello
from games.othello_data import encode_channels

__author__ = 'pliskowski'


def create_dnn_agent(model_dir, model_name, input_size=8, channels=2, batch_size=1, model_iter=30000, gpu=1):
    net = load_net(join(model_dir, model_name), model_name, input_size, batch_size, channels, model_iter, gpu)
    # net has learned transformed labels, so get the encoder
    encoder = joblib.load(join(model_dir, model_name, 'dataset', 'encoder.pkl'))
    
    def dnn_agent(board, my_color):
        probabilities = predict(net, [encode_channels(board[1:-1, 1:-1], my_color)])
        prediction = np.argmax(probabilities)
        move = encoder.inverse_transform(prediction)
        return othello.decode_move(move)
    return dnn_agent
