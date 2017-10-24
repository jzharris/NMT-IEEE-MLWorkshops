# Required packages:
# pip3 install matplotlib
# sudo apt-get install python3-tk
# pip3 install scipy

# Plot ad hoc CIFAR10 instances
import numpy
from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# create a grid of 3x3 images
for i in range(0, 9):
	pyplot.subplot(330 + 1 + i)
	pyplot.imshow(toimage(X_train[i]))
# show the plot
pyplot.show()