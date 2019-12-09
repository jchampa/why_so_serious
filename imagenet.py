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



def buildImageNet(num_classes, mobileBoolean):
	weights = 'imagenet'
	input_tensor = tf.keras.layers.Input(shape=(480,640,3))
	img_shape = [480,640,3]

	if(mobileBoolean):
		base_model = MobileNetV2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=img_shape)
	else:
		base_model = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=img_shape)
	

	#Don't train established layers
	for layer in base_model.layers:
		layer.trainable=False

	 # Add final layers that we train
	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Flatten()(x)
	x = tf.keras.layers.Dense(1024, activation='relu')(x)
	x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

	 # This is the model we will train
	model = Model(base_model.input, x)

	return model

def main():


	num_classes = 7
	learnrate = .01
	batchsz = 128

	#Make sure to run convertGreyscaleInputsToRGB.py first to convert inputs to 3rd channel
	
	inputs, labels = load_data('rgbInputs3.npy', 'rgbLabels3.npy')

	#Set to false to train with RESNET50
	trainWithMobileNetV2 = True
	model = buildImageNet(num_classes, trainWithMobileNetV2)

	adam = tf.keras.optimizers.Adam(learning_rate=learnrate, beta_1=0.9, beta_2=0.999, amsgrad=False)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
	
	X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.1, random_state=42)
	model.fit(X_train, y_train,
		epochs=5,
		batch_size=batchsz)
	score, evaluateAcc = model.evaluate(X_test, y_test, batch_size=128)
	

	print('+++++++++++++++++++++++++')
	print("Test Accuracy: ", evaluateAcc)

	
	

def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

if __name__ == '__main__':
	main()
