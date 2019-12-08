import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, Reshape, Input, ZeroPadding2D, Activation, MaxPooling2D, Add, AveragePooling2D
from tensorflow.keras.models import Model, load_model

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
    skip = Conv2D(filter3, (1, 1), strides = (3,3))(skip)
    skip = BatchNormalization()(skip)

    input = Conv2D(filter1, (1, 1), strides = (3,3))(input)
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

def build_res_cnn(input_shape = (460, 640 ,1)):
    inp = Input(input_shape)

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
    actor = Dense(7, activation='softmax')(actor)
    model = Model(inputs = inp, outputs = actor)

    return model

class RES_CNN(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        # TODO: Initialize all hyperparameters
        self.batch_size = 32
        self.num_classes = 7
        self.epochs = 5
        self.num_folds = 10
        self.epsilon = .001
        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


        # self.model = tf.keras.Sequential()
        #
        # #conv1
        # self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu'))
        #
        # #conv2
        # self.model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu'))
        # self.model.add(tf.keras.layers.BatchNormalization(epsilon = self.epsilon))
        #
        # #conv3
        # self.model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(2,2), padding='SAME', activation='relu'))
        # self.model.add(tf.keras.layers.BatchNormalization(epsilon = self.epsilon))
        #
        # self.model.add(tf.keras.layers.Flatten())
        #
        # self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        # self.model.add(tf.keras.layers.Dense(units=self.num_classes, activation='softmax'))
        self.model = build_res_cnn()
#       # print(self.model.summary())



    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        return self.model(inputs)

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))


    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# model = RES_CNN()
