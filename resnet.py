from __future__ import absolute_import
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2




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
	prob = model(test_inputs)
	return accuracy(prob, test_labels)


def buildResNet(num_classes):
	weights = 'imagenet'
	input_tensor = tf.keras.layers.Input(shape=(640,480,1))
	# img_shape = [640,480,1]
	base_model = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False)
	# base_model = MobileNetV2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=(640,480,1))

	#Don't train established layers
	for layer in base_model.layers:
		layer.trainable=False

	 # Add final layers
	x = base_model.output
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

	 # This is the model we will train
	model = Model(base_model.input, x)

	return model

def main():


	num_classes = 7
	epsilon = .001
	learnrate = .01
	batchsz = 10
	
	inputs, labels = load_data('inputs3.npy', 'labels3.npy')
	new_inputs = []
	new_labels = []

	#fix class imbalance
	for i in range(len(labels)):
		if(labels[i]==0):
			#randomly skip
			randInt = random.randint(1,3)
			if(randInt!=3):
				continue
		new_inputs.append(inputs[i])
		new_labels.append(labels[i])
	new_inputs = (np.asarray(new_inputs)/255.0).astype(np.float32)
	new_labels = np.asarray(new_labels)

	
	model = buildResNet(num_classes)

	adam = tf.keras.optimizers.Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	X_train, X_test, y_train, y_test = train_test_split(new_inputs, new_labels, test_size=0.1, random_state=42)
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
	
	

def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

if __name__ == '__main__':
	main()
