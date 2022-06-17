"""
    pathmind.com:

    neptune.ai:
        Stylegan architecture:

"""
from tensorflow import keras
from keras import optimizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical


"""
    Attempt 2
"""
import matplotlib.pyplot as plot
import numpy as np

from keras.datasets import mnist
from keras.layers import Dense, Dropout, \
Activation, Flatten, BatchNormalization, Reshape
from keras.layers.advanced_activations import LeakyReLU

from keras.layers.convolutional import Conv2D, Conv2DTranspose

from keras.models import Sequential
from keras.optimizers import Adam

dim = 1024 #assuming square images.
channels = 1 #3?

img_shape = (dim, dim, channels)

def gen_cnn(z_shape):
    model = Sequential()

    model.add(Dense(12544, input_dim = z_shape))
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranpose(64, kernel_size=3, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.01))

    model.add(Conv2DTranspose(1, kernel_size=3, strides=2, padding='same'))
    model.add(Activation('tanh'))

    #opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    return model
