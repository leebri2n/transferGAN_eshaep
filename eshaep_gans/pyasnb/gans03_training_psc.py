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

def gen_cnn():
    model = Sequential()

    model.add(Conv2D(1024, (2, 2), padding='same', input_shape=(1,4)))
    model.add(Activation('relu'))

    model.add(Conv2D(512, (2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (2,2), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(3, (2,2), padding='same'))
    model.add(Activation('relu'))

    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])
    print(model.summary())

    return model


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
