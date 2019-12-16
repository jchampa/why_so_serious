import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Reshape, Input, ZeroPadding2D, Activation, MaxPooling2D, Add, AveragePooling2D
from tensorflow.keras.models import Model, load_model
#RESNET MODEL
def identity(input, filter1, filter2, filter3):

    skip = input

    input = Conv2D(filter1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    input = Conv2D(filter2, kernel_size = (3, 3), strides = (1,1), padding = 'same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    input = Conv2D(filter3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(input)
    input = BatchNormalization()(input)

    input = Add()([input, skip])
    input = Activation('relu')(input)

    return input


def conv(input, filter1, filter2, filter3):

    skip = input
    skip = Conv2D(filter3, (1, 1), strides = (3,3), padding= 'same')(skip)
    skip = BatchNormalization(axis = 3)(skip)

    input = Conv2D(filter1, (1, 1), strides = (3,3), padding = 'same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    input = Conv2D(filter2, kernel_size=(3, 3), strides=(1, 1), padding='same')(input)
    input = BatchNormalization()(input)
    input = Activation('relu')(input)

    input = Conv2D(filter3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(input)
    input = BatchNormalization()(input)

    input = Add()([input, skip])
    input = Activation('relu')(input)

    return input

def build_res_cnn(input_shape = (480, 640 ,1)):
    inp = Input(input_shape)
    X = conv(inp, 128, 128, 256)
    X = ZeroPadding2D((3, 3))(inp)

    X = Conv2D(64, (7, 7), strides = (2, 2))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)

    X = conv(X, 128, 128, 256)
    X = identity(X, 64, 64, 256)
    X = identity(X, 64, 64, 256)
    X = identity(X, 64, 64, 256)

    X = conv(X, 128, 128, 256)
    X = identity(X, 128, 128, 256)
    X = identity(X, 128, 128, 256)
    X = identity(X, 128, 128, 256)

    X = Flatten()(X)
    actor = Dense(128, activation='relu')(X)
    actor = Dense(64, activation='relu')(actor)
    actor = Dense(8, activation='softmax')(actor)
    model = Model(inputs = inp, outputs = actor)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model
