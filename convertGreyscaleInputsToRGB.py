from __future__ import absolute_import
import math
import os
import tensorflow as tf
import numpy as np
import random


#Takes inputs and labels, reduces class imbalance, and converts to rgb for keras's pre-trained models

def main():

	
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

	print(new_inputs.shape)

	inputsTensor = tf.convert_to_tensor(new_inputs, dtype=None,dtype_hint=None,name=None)

	print(inputsTensor.shape)

	rgbTensor = tf.image.grayscale_to_rgb(inputsTensor,name=None)

	print(rgbTensor.shape)

	rgbNumpy = np.asarray(rgbTensor)

	print(rgbNumpy.shape)

	np.save('rgbInputs3.npy', rgbNumpy, allow_pickle=True)
	np.save('rgbLabels3.npy', new_labels, allow_pickle=True)
	

def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

if __name__ == '__main__':
	main()
