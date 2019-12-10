from __future__ import absolute_import
from preprocess import load_data
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

def get_trained_model(seed=None):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),input_shape=[480, 640, 1], padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))


	#conv2
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	#conv3
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='SAME', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Dense(units=8, activation= 'softmax'))

    model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])

    X_train, X_test, y_train, y_test = get_data()

    model.fit(X_train, y_train, batch_size=32, nb_epoch=100,
	  verbose=1, validation_data=(X_test, y_test))

    model.save('vanillaModel.' + str(seed))
    score = model.evaluate(X_test, y_test, verbose=0)
    print("***test accuracy: {} ***".format(score))
    return model

def get_data():
    inputs = np.load('inputs3.npy', allow_pickle=True)
    labels = np.load('labels3.npy', allow_pickle=True)
    new_inputs = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i]==0:
            randInt = random.randint(1,3)
            if randInt!=3:
                continue
        new_inputs.append(inputs[i])
        new_labels.append(labels[i])
    
    inputs = (np.asarray(new_inputs)/255.0).astype(np.float32)
    labels = np.asarray(new_labels, dtype=np.uint8)

    unique_elements, counts_elements = np.unique(labels, return_counts=True)

    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)
    y_train =tf.keras.utils.to_categorical(y_train)
    y_test =tf.keras.utils.to_categorical(y_test)

    return X_train, X_test, y_train, y_test
    
def get_accuracies(X_train, X_test, y_train, y_test):
    seeds = np.arange(7)
    models = []
    # ---------------------
    # Train
    for seed in seeds:
        # model = get_trained_model(seed=seed)
        model = tf.keras.models.load_model('vanillaModel.' + str(seed))
        models.append(model)
    # ----------------------
    # Test
    y_test = np.argmax(y_test, axis=1)
    
    single_accuracies = []
    for j, model in enumerate(models):
        probabilities = models[j].predict(X_test)
        predictions = np.argmax(probabilities, axis=1)

        # Get single model accuracy
        single_correct_predictions = tf.equal(predictions, y_test)
        single_accuracies.append(tf.reduce_mean(tf.cast(single_correct_predictions, tf.float32)).numpy())

        predictions = np.expand_dims(predictions, axis=1)
        if j == 0:
            all_predicted = predictions
        else:
            all_predicted = np.append(all_predicted, predictions, axis=1)
    
    # Get ensemble accuracy
    all_predicted_labels = []
    for predictions in all_predicted:
        values, counts = np.unique(predictions, return_counts=True)
        ind = np.argmax(counts)
        all_predicted_labels.append(values[ind])

    correct_predictions = tf.equal(all_predicted_labels, y_test)
    ensemble_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)).numpy()

    return ensemble_accuracy, single_accuracies

def main():
    X_train, X_test, y_train, y_test = get_data()
    for i in range(1):
        ensemble_accuracy, single_accuracies = get_accuracies(X_train, X_test, y_train, y_test)
        print('Ensemble:', ensemble_accuracy)
        print('Single Median:', np.median(single_accuracies), 'Single Average:', np.mean(single_accuracies))
        print('Single:', single_accuracies)

if __name__ == '__main__':
    main()
