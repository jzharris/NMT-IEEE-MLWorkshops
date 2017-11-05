# Code taken from video: https://www.youtube.com/watch?v=ftMq5ps503w

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import lstm, time # helper libraries

x_train, y_train, x_test, y_test = lstm.load_data('sp500.csv', 50, True)

