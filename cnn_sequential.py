from __future__ import absolute_import
from preprocess import load_data
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold


class Model(tf.keras.Model):
	def __init__(self):
		"""
		This model class will contain the architecture for your CNN that
		classifies images. Do not modify the constructor, as doing so
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()

		# TODO: Initialize all hyperparameters
		self.batch_size = 64
		self.num_classes = 7
		self.epochs = 5
		self.epsilon = .001
		self.learning_rate = .01
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


		self.model = tf.keras.Sequential()



		#conv1, I read on a stack overflow that the best order is to do batch norm, relu, max pool but who knows if thats right
		self.model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		self.model.add(tf.keras.layers.BatchNormalization(epsilon = self.epsilon))
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))


		#conv2
		self.model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		self.model.add(tf.keras.layers.BatchNormalization(epsilon = self.epsilon))
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

		#conv3
		self.model.add(tf.keras.layers.Conv2D(filters=20, kernel_size=(3,3), strides=(2,2), padding='SAME'))
		self.model.add(tf.keras.layers.BatchNormalization(epsilon = self.epsilon))
		self.model.add(tf.keras.layers.Activation('relu'))
		self.model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

		self.model.add(tf.keras.layers.Flatten())

		self.model.add(tf.keras.layers.Dense(units=100, activation='relu'))
		self.model.add(tf.keras.layers.Dense(units=64, activation='relu'))
		self.model.add(tf.keras.layers.Dropout(0.25))
		self.model.add(tf.keras.layers.Dense(units=self.num_classes, activation='softmax'))
		


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
		correct_predictions = tf.equal(tf.argmax(logits, 1), labels)
		return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
	and labels - ensure that they are shuffled in the same order using tf.gather.
	To increase accuracy, you may want to use tf.image.random_flip_left_right on your
	inputs before doing the forward pass. You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
	'''

	#shuffle inputs
	indices = range(0, len(train_labels))
	tf.random.shuffle(indices)
	train_inputs = tf.gather(train_inputs, indices)
	train_labels = tf.gather(train_labels, indices)

	for i in range(0, len(train_inputs)-model.batch_size, model.batch_size):
		batchInputs = train_inputs[i:i+model.batch_size]
		labels = train_labels[i:i+model.batch_size]
		#add dimension so its (64,1)
		labels = np.expand_dims(labels, axis=1)
		#need to make inputs 4 dimensional
		tf.reshape(batchInputs, [model.batch_size, 640, 480, 1])
		batchInputs = np.expand_dims(batchInputs, axis=3)
		#randomly flip, this brings down accuracy from what I can tell altho accuracy can range from like 9%-27% so who knows
		# tf.image.random_flip_left_right(batchInputs)
		with tf.GradientTape() as tape:
			logits = model.call(batchInputs)
			batchLoss = model.loss(logits, labels)
			print('Batch Loss {}: {}'.format(i, batchLoss))
		gradients = tape.gradient(batchLoss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		print("Training Accuracy: ", model.accuracy(logits, labels))



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
	test_inputs = np.expand_dims(test_inputs, axis=3)
	prob = model.call(test_inputs)
	return model.accuracy(prob, test_labels)


def main():
	inputs, labels = load_data('inputs.npy', 'labels.npy')
	model = Model()
	#need to convert images and labels to numpy array, normalize images, convert to float32
	image = (np.asarray(inputs)/255.0).astype(np.float32)
	labels = np.asarray(labels)
	#train test split
	X_train, X_test, y_train, y_test = train_test_split(image, labels, test_size=0.1, random_state=42)


	#train for num_epochs
	for ep in range(0, model.epochs):
		print('==========EPOCH {}=========='.format(ep+1))

		kf = KFold(n_splits=5)
		splitIDX = 1
		for train_index, test_index in kf.split(X_train):
			k_x_train, k_x_test = X_train[train_index], X_train[test_index]
			k_y_train, k_y_test = y_train[train_index], y_train[test_index]
			train(model, k_x_train, k_y_train)
			k_test_acc = test(model, k_x_test, k_y_test)
			print("*** FOLD {} test accuracy: {} ***".format(splitIDX, k_test_acc))
			splitIDX+=1



	testAcc = test(model, X_test, y_test)
	print("OVERALL TESTING ACCURACY: ", testAcc)




if __name__ == '__main__':
	main()
