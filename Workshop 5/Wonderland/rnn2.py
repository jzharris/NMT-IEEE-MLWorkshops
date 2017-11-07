import os
import os.path as path
import argparse
import sys

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

parser.add_argument('--epochs', default=50)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--model_name', default='wonderland')
parser.add_argument('--seq_length', default=100)
parser.add_argument('--generate', default=False)
args = parser.parse_args()

EPOCHS = int(args.epochs)
model_name = args.model_name
prev_name = args.model_name
batch_size = int(args.batch_size)
seq_length = int(args.seq_length)
generate = bool(args.generate)

char_to_int = None
int_to_char = None
n_vocab = None
dataX = []
dataY = []
########################################################################################################################


def load_data():
    print('--- loading data')

    ####################################################################################################################
    # get most recent weights gen
    global model_name
    w_inc = 0
    for _, _, files in os.walk("weights"):
        w_inc += files.__len__()
    model_name = model_name + "_{}f_{}r_{}".format(2, seq_length, w_inc)
    global prev_name
    prev_name = prev_name + "_{}f_{}r_{}".format(2, seq_length, w_inc - 1)

    ####################################################################################################################
    # load dataset:
    # load ascii text and covert to lowercase
    filename = "wonderland.txt"
    raw_text = open(filename).read()
    raw_text = raw_text.lower()

    # create mapping of unique chars to integers
    chars = sorted(list(set(raw_text)))
    global char_to_int
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    global int_to_char
    int_to_char = int_to_char = dict((i, c) for i, c in enumerate(chars))

    n_chars = len(raw_text)
    global n_vocab
    n_vocab = len(chars)
    print("--- total characters: {}".format(n_chars))
    print("--- total vocab: {}".format(n_vocab))

    # prepare the dataset of input to output pairs encoded as integers
    global dataX
    global dataY
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
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
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
    filename = 'weights/{}_weights.best.hdf5'.format(prev_name)
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # pick a random seed
    start = np.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed: {}\n".format(''.join([int_to_char[value] for value in pattern])))
    print('---')
    # generate characters
    for i in range(1000):
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / float(n_vocab)
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_to_char[index]
        sys.stdout.write(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    print('---')
    print("\nDone.")


def main():
    if not path.exists('out'):
        os.mkdir('out')

    if not path.exists('weights'):
        os.mkdir('weights')

    X, y = load_data()
    model = build_model(X, y)
    if not generate:
        train_model(model, X, y)
    else:
        generate_text(model)

if __name__ == '__main__':
    main()