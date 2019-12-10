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
    # labels = np.load("../labels3.npy")
    labels = np.load("../labels3.npy")

    new_landmarks = []
    new_labels = []

    for i in range(len(labels)):
        if(labels[i]==0):
            randInt = random.randint(1,3)
            if(randInt!=3):
                continue
            
        new_landmarks.append(spacial_landmarks[i].flatten())
        new_labels.append(tf.keras.utils.to_categorical(labels[i],8))
    
    x_train, x_test, y_train, y_test = train_test_split(new_landmarks, new_labels, test_size=0.1, random_state=42)

    x_train,x_test = np.asarray(x_train),np.asarray(x_test)
    y_train,y_test = np.asarray(y_train),np.asarray(y_test)

    print(x_train[0].shape)
    print(y_train[0].shape)


    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, input_shape=(4624,)))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(units=8, activation='softmax'))
    

    model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, nb_epoch=100,verbose=1, validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("***test accuracy: {} ***".format(score))

    
    
    
    


if __name__ == "__main__":
    main()