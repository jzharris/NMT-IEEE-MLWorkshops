import os
import os.path as path
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

########################################################################################################################
# Set ArgumentParser so you can set all variables from terminal
parser = argparse.ArgumentParser(description='Test Arguments')

parser.add_argument('--epochs', default=20)
parser.add_argument('--batch_size', default=128)
parser.add_argument('--model_name', default='romeo')
parser.add_argument('--seq_length', default=100)
args = parser.parse_args()

EPOCHS = int(args.epochs)
model_name = args.model_name
prev_name = args.model_name
batch_size = int(args.batch_size)
seq_length = int(args.seq_length)
########################################################################################################################


def load_data():
    print('--- loading data')

    ####################################################################################################################
    # get most recent weights gen
    global model_name
    w_inc = 0
    for _, _, files in os.walk("weights"):
        w_inc += files.__len__()
    model_name = model_name + "_{}f_{}r_{}".format(1, seq_length, w_inc)
    global prev_name
    prev_name = prev_name + "_{}f_{}r_{}".format(1, seq_length, w_inc - 1)

    ####################################################################################################################
    # load dataset:
    # load ascii text and covert to lowercase
    filename = "romeo_and_juliet_full.txt"
    raw_text = open(filename).read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("--- total characters: {}".format(n_chars))
    print("--- total vocab: {}".format(n_vocab))

    # prepare the dataset of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, n_chars - seq_length, 1):
        seq_in = raw_text[i:i + seq_length]
        seq_out = raw_text[i + seq_length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    n_patterns = len(dataX)
    print("--- total patterns: {}".format(n_patterns))

    # reshape X to be [samples, time steps, features]
    X = np.reshape(dataX, (n_patterns, seq_length, 1))
    # normalize
    X = X / float(n_vocab)
    # one hot encode the output variable
    y = np_utils.to_categorical(dataY)
    return X, y


def build_model(X, y):
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def checkpoint():
    # records checkpoints per EPOCH using keras
    print('--- enabling ModelCheckpoint')
    check_path = 'weights/{}_weights.best.hdf5'.format(model_name)
    return ModelCheckpoint(check_path, monitor='loss', verbose=1, save_best_only=True, mode='min')


def train_model(model, X, y):
    # define the checkpoint
    callbacks_list = [checkpoint()]
    model.fit(X, y, epochs=EPOCHS, batch_size=128, callbacks=callbacks_list)


def generate_text(model):
    # load the network weights
    filename = 'weights/{}_weights.best.hdf5'.format(model_name)
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')



def main():
    if not path.exists('out'):
        os.mkdir('out')

    if not path.exists('weights'):
        os.mkdir('weights')

    X, y = load_data()
    model = build_model(X, y)
    train_model(model, X, y)

if __name__ == '__main__':
    main()