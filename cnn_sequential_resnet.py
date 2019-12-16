from __future__ import absolute_import
from preprocess import load_data
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split
from model import build_res_cnn
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img, save_img
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def main(train = True):
    inputs, labels = load_data('inputs4.npy', 'labels4.npy')
    input = np.load('confusion_matrix1.npy', allow_pickle=True)
    #confusion matrix print out
    for row in input:
        print(row)


    new_inputs = []
    new_labels = []
    for i in range(len(labels)):
        if(labels[i]==0):
            randInt = random.randint(1,7)
            if(randInt<5):
                continue
        new_inputs.append(inputs[i])
        new_labels.append(labels[i])

    inputs = (np.asarray(new_inputs)/255.0).astype(np.float32)
    labels = np.asarray(new_labels)
    # print(np.bincount(labels.astype(np.int64)))
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.35, random_state=42)
    y_train  =tf.keras.utils.to_categorical(y_train)
    y_test  =tf.keras.utils.to_categorical(y_test)


    # print(y_test)
    #test images
    file_path = 'Unknown.jpeg'
    file_path1 = 'scared.jpeg'
    file_path2 = 'sad.jpeg'



    img = load_img(file_path,color_mode="grayscale",target_size=(480,640),interpolation="nearest")
    img1 = img_to_array(img)
    img1 = img1[np.newaxis, ...]/255.0

    img = load_img(file_path1,color_mode="grayscale",target_size=(480,640),interpolation="nearest")
    img2 = img_to_array(img)
    img2 = img2[np.newaxis, ...]/255.0

    img = load_img(file_path2,color_mode="grayscale",target_size=(480,640),interpolation="nearest")
    img3 = img_to_array(img)
    img3 = img3[np.newaxis, ...]/255.0

    if train == True:
        model.fit(X_train, y_train, batch_size=32, epochs=50,verbose=1, validation_data=(X_test, y_test))
    # print('saving model')
    # model.save('resnet.h5')
    else:
        model = load_model('resnet.h5')
        print('happy 5')
        print(model(img1))
        print('scared 4')
        print(model(img2))
        print('sad 6')
        print(model(img3))

        # train(model, X_train, y_train)
        # k_test_acc = test(model, X_test, y_test)
        y_pred = model.predict(X_test)
        y_pred = tf.math.argmax(y_pred,axis=1).numpy()


        #confusion matrix build
        y_test = np.argmax(y_test, axis= 1)
        # print(y_test)
        # y_pred = tf.keras.utils.to_categorical(y_pred)

        cm = [[0 for _ in range(8)] for d in range(8)]
        for i in range(len(y_pred)):

            cm[y_test[i]][y_pred[i]] += 1

        cm = np.asarray(cm)
        np.save('confusion_matrix2.npy', cm)






if __name__ == '__main__':
	main(False)
