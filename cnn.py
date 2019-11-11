from __future__ import absolute_import
from preprocess import get_data
import math
import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
	def __init__(self):
		"""
        This model class will contain the architecture for your CNN that
		classifies images. Do not modify the constructor, as doing so
		will break the autograder. We have left in variables in the constructor
		for you to fill out, but you are welcome to change them if you'd like.
		"""
		super(Model, self).__init__()

		self.batch_size = 64
		self.num_classes = 7
        self.epochs = 10
		# TODO: Initialize all hyperparameters
		self.epsilon = .001
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
		# TODO: Initialize all trainable parameters
		self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
		self.fb1 = tf.Variable(tf.random.truncated_normal([16], stddev=0.1))
		self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
		self.fb2 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))
		self.filter3 = tf.Variable(tf.random.truncated_normal([5,5,20,20], stddev=0.1))
		self.fb3 = tf.Variable(tf.random.truncated_normal([20], stddev=0.1))

		self.dw1 = tf.Variable(tf.random.truncated_normal(shape=[4 * 4 * 20,64],stddev=0.1),dtype=tf.float32)
		self.dw2 = tf.Variable(tf.random.truncated_normal(shape=[64,32],stddev=0.1),dtype=tf.float32)
		self.dw3 = tf.Variable(tf.random.truncated_normal(shape=[32,self.num_classes],stddev=0.1),dtype=tf.float32)

		self.db1 = tf.Variable(tf.random.truncated_normal(shape=[64],stddev=0.1),dtype=tf.float32)
		self.db2 = tf.Variable(tf.random.truncated_normal(shape=[32],stddev=0.1),dtype=tf.float32)
		self.db3 = tf.Variable(tf.random.truncated_normal(shape=[1, self.num_classes],stddev=0.1),dtype=tf.float32)


	def call(self, inputs):
		"""
		Runs a forward pass on an input batch of images.
		:param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
		:param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
		:return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
		"""
		# Remember that
		# shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
		# shape of filter = (filter_height, filter_width, in_channels, out_channels)
		# shape of strides = (batch_stride, height_stride, width_stride, channels_stride)
		#convolution 1
		conv = tf.nn.conv2d(inputs, self.filter1, [1, 2,2 ,1], "SAME")
		conv_with_bias = tf.nn.bias_add(conv, self.fb1)
		batch_mean_1, batch_var_1 = tf.nn.moments(conv_with_bias, [0, 1, 2])
		batch_norm_1 = tf.nn.relu(tf.nn.batch_normalization(conv_with_bias,batch_mean_1,batch_var_1,None,None,self.epsilon))
		pooled_conv_1 = tf.nn.max_pool(batch_norm_1,[3,3], [1, 2, 2, 1],"SAME")
		#convolution 2
		conv2 = tf.nn.conv2d(pooled_conv_1, self.filter2, [1, 1,1 ,1], "SAME")
		conv2_with_bias = tf.nn.bias_add(conv2, self.fb2)
		batch_mean_2, batch_var_2 = tf.nn.moments(conv2_with_bias, [0, 1, 2])
		batch_norm_2 = tf.nn.relu(tf.nn.batch_normalization(conv2_with_bias,batch_mean_2,batch_var_2,None,None,self.epsilon))
		pooled_conv_2 = tf.nn.max_pool(batch_norm_2,[2,2], [1, 2, 2, 1],"SAME")

		#convolution 3

		conv3 = tf.nn.conv2d(pooled_conv_2, self.filter3, [1, 1,1 ,1], "SAME")
		conv3_with_bias = tf.nn.bias_add(conv3, self.fb3)
		batch_mean_3, batch_var_3 = tf.nn.moments(conv3_with_bias, [0, 1, 2])
		batch_norm_3 = tf.nn.relu(tf.nn.batch_normalization(conv3_with_bias,batch_mean_3,batch_var_3,None,None,self.epsilon))
		flattened = tf.reshape(batch_norm_3, [-1, 320])
		#dense layers
		layer1Output = tf.nn.dropout(tf.nn.relu(tf.matmul(flattened, self.dw1) + self.db1), .3) # remember to use a relu activation
		layer2Output = tf.nn.dropout(tf.nn.relu(tf.matmul(layer1Output, self.dw2) + self.db2), .3)
		logits = tf.matmul(layer2Output, self.dw3) + self.db3

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


	# Instantiate our model
	indices = tf.random.shuffle(tf.range(len(train_inputs)))
	train_input = tf.gather(train_inputs, indices)
	train_label = tf.gather(train_labels, indices)

	# Choosing our optimizer
	optimizer = model.optimizer #tf.keras.optimizers.Adam(learning_rate=0.001)
	for i in range(int(math.floor(len(train_input)/model.batch_size))):

		input = tf.image.random_flip_left_right(train_input[i*model.batch_size:i * model.batch_size + model.batch_size])
		label = train_label[i*model.batch_size:i * model.batch_size + model.batch_size]


		with tf.GradientTape() as tape:
			predictions = model.call(input)
			loss = model.loss(predictions, label)
			gradients = tape.gradient(loss, model.trainable_variables)
		    optimizer.apply_gradients(zip(gradients, model.trainable_variables))




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

	inputs, labels = get_data('/Data PATH')
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
        if j == 0 or j == group_num -1:
            if j == 0:
                train_inputs = inputs[group_size * (j+1):]
                train_labels = labels[group_size * (j+1):]
            else:
                train_inputs = inputs[:group_size * j]
                train_labels = labels[:group_size * j]
        else:
            train_inputs = inputs[:group_size * j] + inputs[group_size * (j+1)]
            train_labels = labels[:group_size * j] + labels[group_size * (j+1)]

        test_inputs = inputs[group_size * j: group_size * (j+1)]
        test_labels = labels[group_size * j: group_size * (j+1)]

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
