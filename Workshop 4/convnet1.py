# Open up your favorite text editor
# Using MNIST dataset with cutomize network model


import keras
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
print(y_train.shape)

# conda install matplotlib
from matplotlib import pyplot

# display an image...

# create a grid of 3x3 images
for i in range(0, 9):
    ax = pyplot.subplot(330 + 1 + i)
    ax.title.set_text(y_train[i])
    pyplot.imshow(X_train[i], cmap='gray')

# show the plot...
pyplot.show()


from keras.models import Sequential
model = Sequential()
from keras.layers.convolutional import Conv2D
model.add(Conv2D(42, (3, 3), input_shape=[28, 28, 1], activation='relu', padding='same'))
from keras.layers import Flatten
model.add(Flatten())
from keras.layers import Dense
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model.fit(X_train, y_train,
          batch_size=100,
          epochs=1,
          verbose=1,
          validation_data=(X_test, y_test))