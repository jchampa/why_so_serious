from __future__ import absolute_import
from get_landmarks import get_landmarks,calc_landmark_spacial
from sklearn.model_selection import train_test_split
import math
import os
import tensorflow as tf
import numpy as np
import random
import sys


def main():
    if len(sys.argv) > 2:
        print("Invalid args usage: python landmark_cnn.py [PREPROCESS]")
        return
    if len(sys.argv) == 2:
        if sys.argv[1] == "PREPROCESS":
            # inputs = np.load("../inputs3.npy", allow_pickle=True)
            # labels = np.load("../labels3.npy", allow_pickle=True)
            # get_landmarks(inputs,labels)
            landmarks = np.load("landmarks.npy")
            calc_landmark_spacial(landmarks)
        else:
            print("Invalid args usage: python landmark_cnn.py [PREPROCESS]")
            return

    # images,labels,landmarks,one_hot_labels = np.load("../inputs3.npy"),np.load("../labels3.npy"),np.load("landmarks.npy"),np.load("one_hot_labels.npy")
    spacial_landmarks = np.load("landmark_spacial_info.npy")
    labels = np.load("../labels3.npy")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[68,68]))
    model.add(tf.keras.layers.Dense(512, input_shape=(4624,)))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(8, activation='softmax'))

    X_train, X_test, y_train, y_test = train_test_split(spacial_landmarks, labels, test_size=0.1, random_state=42)

    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    unique_elements, counts_elements = np.unique(y_test, return_counts=True)

    opt = tf.keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=32, nb_epoch=100,verbose=1, validation_data=(X_test, y_test))
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print("***test accuracy: {} ***".format(score))

    
    
    
    


if __name__ == "__main__":
    main()