import os
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import random


##THIS FILE IS FOR AUGMENTING THE DATA FOR BINARY CLASSIFICATION BETWEEN HAPPY AND NEUTRAL, REMOVES ABOUT 
##1/3 OF THE NEUTRAL DATA AND THEN AUGMENTS BY 3 FOR THE HAPPY DATA


def augment_images(images,labels,num_augments):
	augmented_images = []
	augmented_labels = []
	for image,label in zip(images,labels):
		augmented_images.append(image[:,:,0])
		augmented_labels.append(label)

		aug_image_array = image[...]

		augment_base = image.reshape((1,)+aug_image_array.shape)
		augment(augment_base,label,augmented_images,augmented_labels,num_augments)
	
	# print("augmented_images[1].shape: ", augmented_images[1].shape)

	# augmented_images = np.squeeze(augmented_images)
	# augmented_labels = np.squeeze(augmented_labels)

	augmented_images = np.asarray(augmented_images)
	augmented_labels = np.asarray(augmented_labels)

	# print("IAMGE SHAOE: ", augmented_images.shape)


	return augmented_images, augmented_labels

	# np.save("augmented_inputs3.npy",augmented_images,allow_pickle=True)
	# np.save("augmented_labels3.npy",augmented_labels,allow_pickle=True)


def augment(start, label, images, labels, augment_number):
	datagen = ImageDataGenerator(
				rotation_range=40,
				width_shift_range=0.2,
				height_shift_range=0.2,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True,
				fill_mode='nearest')
	count = 0

	for batch in datagen.flow(start, batch_size=1,save_to_dir = None, save_prefix='cat', save_format='jpeg'):
		new_image = batch[0,:,:,0]
		images.append(new_image)
		labels.append(label)
		count+=1
		if count == augment_number:
			break



def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

def main():

	#Loads in data from preprocess3.py
	unaugmented_inputs, unaugmented_labels = load_data('inputs3.npy', 'labels3.npy')

	neutral_inputs = []
	neutral_labels = []

	happy_inputs = []
	happy_labels = []

	for i in range(len(unaugmented_labels)):
		if(unaugmented_labels[i]==0):
			#randomly skip 1/3 of neutral data
			randInt = random.randint(1,3)
			if(randInt!=3):
				continue
			neutral_inputs.append(unaugmented_inputs[i][:,:,0])
			neutral_labels.append(unaugmented_labels[i])
		elif(unaugmented_labels[i]==6):
			happy_inputs.append(unaugmented_inputs[i])
			happy_labels.append(unaugmented_labels[i])

	neutral_inputs = np.asarray(neutral_inputs)
	neutral_labels = np.asarray(neutral_labels)
	happy_inputs = np.asarray(happy_inputs)
	happy_labels = np.asarray(happy_labels)
	
	print("length happy_inputs before augment: ", len(happy_inputs))

	tf.reshape(happy_inputs, [happy_inputs.shape[0], 640, 480])
	print("Happy_inputs shape: ", happy_inputs.shape)
	happy_inputs, happy_labels = augment_images(happy_inputs,happy_labels,3)

	print("length happy_inputs after augment: ", len(happy_inputs))
	print("length neutral_inputs: ", len(neutral_inputs))

	print("shape happy_inputs after augment: ", happy_inputs.shape)
	print("shape neutral_inputs: ", neutral_inputs.shape)

	augmented_images = np.concatenate((happy_inputs, neutral_inputs), axis=0)
	print("augmented_images.shape: ", augmented_images.shape)
	augmented_labels = np.concatenate((happy_labels, neutral_labels), axis=0)
	print("augmented_labels.shape: ", augmented_labels.shape)


	np.save("binary_augment_inputs3.npy",augmented_images,allow_pickle=True)
	np.save("binary_augment_labels3.npy",augmented_labels,allow_pickle=True)



if __name__ == '__main__':
	main()