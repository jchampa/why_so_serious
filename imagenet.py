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

from tensorflow.keras.preprocessing.image import array_to_img, save_img




def buildImageNet(num_classes, mobileBoolean, finetuneBoolean):
	weights = 'imagenet'
	input_tensor = tf.keras.layers.Input(shape=(480,640,3))
	img_shape = [480,640,3]

	if(mobileBoolean):
		base_model = MobileNetV2(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=img_shape)
	else:
		base_model = ResNet50(weights=weights, input_tensor=input_tensor, include_top=False, input_shape=img_shape)
	

	if(finetuneBoolean):
		#Don't train established layers, but finetune by training 5
		for layer in base_model.layers[:-5]:
			layer.trainable=False
	else:
		#Don't train ANY established layers
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
	print(model.summary())

	return model

def main():


	num_classes = 8
	batchsz = 32
	numEpoch = 25

	#Make sure to run convertGreyscaleInputsToRGB.py first to convert inputs to 3rd channel
	
	inputs, labels = load_data('rgbInputs3.npy', 'rgbLabels3.npy')

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
	inputs = (np.asarray(new_inputs)/255.0).astype(np.float32)
	labels = np.asarray(new_labels, dtype=np.uint8)



	unique_elements, counts_elements = np.unique(labels, return_counts=True)
	print("labels count")
	print(unique_elements)
	print(counts_elements)



	#Set to false to train with RESNET50
	trainWithMobileNetV2 = True
	#Set to True to train finetune and train last few layers of imagenet
	finetuneSomeLayers = False
	model = buildImageNet(num_classes, trainWithMobileNetV2, finetuneSomeLayers)
	print(model.summary())

	model.compile(loss='categorical_crossentropy',
				  optimizer='adam',
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


	model.fit(X_train, y_train, batch_size=batchsz, nb_epoch=numEpoch,
	  verbose=1, validation_data=(X_test, y_test))

	score = model.evaluate(X_test, y_test, verbose=0)
	print("***test accuracy: {} ***".format(score))


	

	
	

def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

if __name__ == '__main__':
	main()
