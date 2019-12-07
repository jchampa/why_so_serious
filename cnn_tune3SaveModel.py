from __future__ import absolute_import
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split



def accuracy(logits, labels):
	"""
	Calculates the model's prediction accuracy by comparing
	logits to correct labels
	:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

	:return: the accuracy of the model as a Tensor
	"""
	correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))




def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels. You should NOT randomly
	flip images or do any extra preprocessing.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	tf.reshape(test_inputs, [test_inputs.shape[0], 640, 480, 1])
	# test_inputs = np.expand_dims(test_inputs, axis=3)
	prob = model(test_inputs)
	return accuracy(prob, test_labels)


def main():

	l = [[0.015 ,     30        ],
	[0.002     , 40        ],
	[0.01      , 40        ],
	[0.08      , 50        ],
	[0.025     , 60        ],
	[0.06      , 60       ],
	[0.1       , 70       ],
	[0.015     , 80       ],
	[0.06      , 80       ],
	[0.002     , 90       ],
	[0.005     , 90       ],
	[0.01      , 90       ],
	[0.02      , 90      ],
	[0.06      , 90      ]]

	num_classes = 7
	epsilon = .001
	
	
	inputs, labels = load_data('inputs3.npy', 'labels3.npy')
	for tup in l:
		learnrate = tup[0]
		batchsz = tup[1]
		model = tf.keras.Sequential()
		#conv1
		model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		model.add(tf.keras.layers.BatchNormalization(epsilon = epsilon))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		#conv2
		model.add(tf.keras.layers.Conv2D(filters=40, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		model.add(tf.keras.layers.BatchNormalization(epsilon = epsilon))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		#conv3
		model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		model.add(tf.keras.layers.BatchNormalization(epsilon = epsilon))
		model.add(tf.keras.layers.Activation('relu'))
		model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(units=100, activation='relu'))
		model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		model.add(tf.keras.layers.Dropout(0.25))
		model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

		adam = tf.keras.optimizers.Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
		model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
		image = (np.asarray(inputs)/255.0).astype(np.float32)
		labels = np.asarray(labels)
		X_train, X_test, y_train, y_test = train_test_split(image, labels, test_size=0.1, random_state=42)
		model.fit(X_train, y_train,
		  epochs=5,
		  batch_size=batchsz)
		score, evaluateAcc = model.evaluate(X_test, y_test, batch_size=128)
		
		testAcc = test(model, X_test, y_test)

		print('+++++++++++++++++++++++++')
		print("Learning Rate: ", learnrate)
		print("Batch Size: ", batchsz)
		print("OVERALL TESTING ACCURACY: ", testAcc)
		print("ACCC: ", evaluateAcc)
		if(testAcc>.45):
			model.save('goodModel.h5')
			print('+++++++++++++++++++++++++')
			print("Learning Rate: ", learnrate)
			print("Batch Size: ", batchsz)
			print("OVERALL TESTING ACCURACY: ", testAcc)
			return



	
def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

if __name__ == '__main__':
	main()
