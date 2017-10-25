## Example is from here:
## https://machinelearningmastery.com/object-recognition-convolutional-neural-networks-keras-deep-learning-library/

# Simple CNN model for CIFAR-10

import os
import os.path as path

import numpy
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard

# Compile model
epochs = 50
lrate = 0.01

# Simply orders the dataset for 'channels (3) first'
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


def load_data():
    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return X_train, y_train, X_test, y_test


def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def tensorboard():
    # starting tensorboard: tensorboard --logdir=run1:logs/ --port 6006
    if not path.exists('logs'):
        os.mkdir('logs')
    print('--- enabling TensorBoard')
    return TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=True)


def train(model, X_train, y_train, X_test, y_test):
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64, callbacks=[tensorboard()])

    return model


def evaluate(model, X_test, y_test):
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))


def load(model, filename):
    if not path.exists('out'):
        os.mkdir('out')

    if path.exists(filename):
        model.load_weights(filename)

    return model


def save(model, filename):
    if not path.exists('out'):
        os.mkdir('out')

    model.save_weights(filename)


def main():
    filename = 'out/cifar100_complex_cnn.h5'

    X_train, y_train, X_test, y_test = load_data()

    model = load(build_model(y_test.shape[1]), filename)

    model = train(model, X_train, y_train, X_test, y_test)

    evaluate(model, X_test, y_test)

    save(model, filename)


if __name__ == '__main__':
    main()