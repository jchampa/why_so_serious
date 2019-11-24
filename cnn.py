from __future__ import absolute_import
from preprocess import get_data, load_data
import math
import os
import tensorflow as tf
import numpy as np
import random

# from sklearn.model_selection import KFold
#OlD VERSION

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
		self.epochs = 10
		self.num_folds = 10
		self.epsilon = .001
		self.learning_rate = .001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)


		# TODO: Initialize all trainable parameters

		#filters and filter biases
		self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
		self.fb1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))

		self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
		self.fb2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

		self.filter3 = tf.Variable(tf.random.truncated_normal([5,5,20,20], stddev=0.1))
		self.fb3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

		#Dense layers
		self.dense1 = tf.keras.layers.Dense(64, activation='relu')
		self.dropout1 = tf.keras.layers.Dropout(rate=.3, noise_shape=None, seed=None)
		self.dense2 = tf.keras.layers.Dense(64, activation='relu')
		self.dropout2 = tf.keras.layers.Dropout(rate=.3, noise_shape=None, seed=None)
		self.dense3 = tf.keras.layers.Dense(self.num_classes, activation='softmax')

	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""

		num_inputs = tf.shape(inputs)[0]

		stride2 = [1, 2, 2, 1]
		stride1 = [1, 1, 1, 1]

		# Convolution Layer 1
		conv1 = tf.nn.conv2d(inputs, self.filter1, stride2, padding="SAME")
		conv1 = tf.nn.bias_add(conv1, self.fb1)
		# Batch Normalization 1
		mean, variance = tf.nn.moments(conv1, [0, 1, 2])
		batch1 = tf.nn.batch_normalization(conv1, mean, variance, None, None, self.epsilon)
		# ReLU Nonlinearlity 1
		relu1 = tf.nn.relu(batch1)
		# Max Pooling 1
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=stride2, padding="SAME")

		# Convolution Layer 2
		conv2 = tf.nn.conv2d(pool1, filters=self.filter2, strides=stride1, padding="SAME")
		conv2 = tf.nn.bias_add(conv2, self.fb2)
		# Batch Normalization 2
		mean, variance = tf.nn.moments(conv2, [0, 1, 2])
		batch2 = tf.nn.batch_normalization(conv2, mean, variance, None, None, self.epsilon)
		# ReLU Nonlinearlity 2
		relu2 = tf.nn.relu(batch2)
		# Max Pooling 2
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=stride2, padding="SAME")

		# Convolution Layer 3
		conv3 = tf.nn.conv2d(pool2, filters=self.filter3, strides=stride1, padding="SAME")
		conv3 = tf.nn.bias_add(conv3, self.fb3)
		# Batch Normalization 3
		mean, variance = tf.nn.moments(conv3, [0, 1, 2])
		batch3 = tf.nn.batch_normalization(conv3, mean, variance, None, None, self.epsilon)
		# ReLU Nonlinearlity 3
		relu3 = tf.nn.relu(batch3)
		#reshape
		shape = tf.shape(relu3)
		relu3 = tf.reshape(relu3, [num_inputs, shape[1]*shape[2]*shape[3]])

		# Dense Layer 1
		d1 = dense1(relu3)
		d1 = dropout1(d1)
		# Dense Layer 2
		d2 = dense2(d1)
		d2 = dropout2(d2)
		# Dense Layer 3
		logits = dense3(d2)

		return logits

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

	optimizer = model.optimizer
	#shuffle inputs
	indices = tf.random.shuffle(tf.range(len(train_inputs)))
	train_input = tf.gather(train_inputs, indices)
	train_label = tf.gather(train_labels, indices)


	for i in range(len(train_input)//model.batch_size):
		input = train_input[i*model.batch_size:i * model.batch_size + model.batch_size]
		tf.image.random_flip_left_right(input)
		label = train_label[i*model.batch_size:i * model.batch_size + model.batch_size]
		with tf.GradientTape() as tape:
			logits = model.call(input)
			loss = model.loss(predictions, label)
			print('Batch Loss: ', loss)
		gradients = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(gradients, model.trainable_variables))
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
	prob = model.call(test_inputs, True)
	return model.accuracy(prob, test_labels)


def main():

	inputs, labels = load_data('inputs.npy', 'labels.npy')
	group_num = 10
	group_size = len(inputs) // group_num
	acc = 0
	for j in range(group_num):
        # dont want to do this until it actually starts working...
		# for cross validation
		if j == 1:
			break
		print('test set is: {} out of {}'.format(j, group_num))
		model = Model()
		# create train/test sets by excluding the current test set

		if j == 0:
			train_inputs = inputs[group_size * (j+1):]
			train_labels = labels[group_size * (j+1):]
		elif j == group_num - 1:
			train_inputs = inputs[:group_size * j]
			train_labels = labels[:group_size * j]
		else:
			train_inputs = inputs[:group_size * j] + inputs[group_size * (j+1):]
			train_labels = labels[:group_size * j] + labels[group_size * (j+1):]

		test_inputs = inputs[group_size * j: group_size * (j+1)]
		test_labels = labels[group_size * j: group_size * (j+1)]
		print(train_inputs.shape)
		for i in range(model.epochs):
			print('epoch #{}'.format(i+1))
			train(model, train_inputs, train_labels)
			prob = model.call(train_inputs)
			acc = model.accuracy(prob, train_labels)
			print('train acc: {}'.format(acc))
			prob = model.call(test_inputs)
			acc = model.accuracy(prob, test_labels)
			print('test acc: {}'.format(acc))
		curr_acc = test(model, test_inputs, test_labels)
		acc += curr_acc
		print('test group {} acc: {}'.format(j, curr_acc))
	# acc = test(model, test_inputs, test_labels)
	# print('accuracy of model: {}'.format(acc))

    # overall_acc = acc / float(group_num)
	overall_acc = acc
	print('overall acc: {}'.format(overall_acc))

	return


if __name__ == '__main__':
	main()
