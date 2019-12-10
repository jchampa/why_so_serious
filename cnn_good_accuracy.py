from __future__ import absolute_import
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import Model, load_model




def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels



def main():

	#Load in inputs
	inputs, labels = load_data('inputs3.npy', 'labels3.npy')

	#Further fix class imbalance
	new_inputs = []
	new_labels = []
	for i in range(len(labels)):
		if(labels[i]==0):
			#randomly skip
			# continue
			randInt = random.randint(1,3)
			if(randInt!=3):
				continue
		new_inputs.append(inputs[i])
		new_labels.append(labels[i])

	#convert images and labels to numpy array, normalize images, convert to float32
	inputs = (np.asarray(new_inputs)/255.0).astype(np.float32)
	labels = np.asarray(new_labels, dtype=np.uint8)

	unique_elements, counts_elements = np.unique(labels, return_counts=True)
	print("all labels count")
	print(unique_elements)
	print(counts_elements)




	# model = Model()
	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2),input_shape=[480, 640, 1], padding='SAME'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))


	#conv2
	model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='SAME'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	#conv3
	model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='SAME'))
	model.add(tf.keras.layers.BatchNormalization())
	model.add(tf.keras.layers.Activation('relu'))
	# self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

	model.add(tf.keras.layers.Flatten())

	model.add(tf.keras.layers.Dense(units=256, activation='relu'))
	model.add(tf.keras.layers.Dense(units=128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Dense(units=8, activation= 'softmax'))
	#compiling the model
	print(model.summary())


	useAdam = True
	if(useAdam):
		# adam = tf.keras.optimizers.Adam(learning_rate=.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
		model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
				  metrics=['accuracy'])
	else:
		model.compile(loss='categorical_crossentropy',
					optimizer='adadelta',
					metrics=['accuracy'])

	#train test split
	X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)

	unique_elements, counts_elements = np.unique(y_train, return_counts=True)
	print("Y Train counts")
	print(unique_elements)
	print(counts_elements)

	unique_elements, counts_elements = np.unique(y_test, return_counts=True)
	print("Y Test counts")
	print(unique_elements)
	print(counts_elements)



	#Turn labels to categorical for loss function
	y_train =tf.keras.utils.to_categorical(y_train)
	y_test =tf.keras.utils.to_categorical(y_test)


	model.fit(X_train, y_train, batch_size=32, nb_epoch=100,
	  verbose=1, validation_data=(X_test, y_test))
	score = model.evaluate(X_test, y_test, verbose=0)
	print("***test accuracy: {} ***".format(score))


	model.save('vanillaModel.h5')
	




if __name__ == '__main__':
	main()