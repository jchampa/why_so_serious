import os
import imageio
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
def get_data(images_path, labels_path):
	'''
	Note that num images per subject varies.
	Also not every sequence has a label.
	:param images_path:
	:param labels_path:
	:return: data_dictionary(imageID) -> [image as a numpy array, label as float]
			 imageID created using path
	'''
	data_dictionary = {}

	print(images_path, labels_path)
	# Read in images.
	intense_num = {}
	counter = 0
	rgb = set()
	for subdir, dirs, files in os.walk(images_path):

		for file in files:

			if file == '.DS_Store':
				continue


			file_path = os.path.join(subdir, file)
			key = subdir.replace(images_path, '')
			num = int(file_path[-6:-4])
			if key in intense_num:
				if intense_num[key] < num:
					intense_num[key] = num
			else:
				intense_num[key] = num
			# image = cv2.resize(cv2.imread(file_path, 0), (480, 640))
			# images = []
			image = img_to_array(load_img(file_path,color_mode="grayscale",target_size=(480,640),interpolation="nearest"))

			# augment_images(augment_base,images,3)

			# images.append(np.reshape(image,(480,640)))

			image = np.reshape(image,(480,640))
			
			if key not in data_dictionary:
				data_dictionary[key] = {}
			# if len(image.shape) == 3:
			# 	#convert to greyscale
			# 	#dont know if this is correct way to do it but...
			# 	rgb.add(key)
			#
			# 	image = np.dot(image[...,:3], [0.299, 0.587, 0.114])
			data_dictionary[key][num] = {'image': image, 'label': None}
			# data_dictionary[key] = [1]

	# Read in numbers from labels.


	counter = 0
	for subdir, dirs, files in os.walk(labels_path):
		for file in files:

			if file == '.DS_Store':
				continue

			file_path = os.path.join(subdir, file)
			num = int(file_path[-15:-12])
			key = subdir.replace(labels_path, '')[1:] # remove first / to have same key as images
			with open(file_path, 'r') as f:
				# Remove new line character at end and cast as float
				label = float(f.read().rstrip('\n'))
				if key in data_dictionary:
					for k in data_dictionary[key]:
						if k == 1:
							data_dictionary[key][k]['label'] = 0
						else:
							data_dictionary[key][k]['label'] = label

	inputs = []
	labels = []
	# Remove images and labels with no label.
	checker = []
	for key in data_dictionary:
		#this is if we only want to do first and most intense image in the folder
		k = intense_num[key]
		current_input = data_dictionary[key][k]['image']
		current_label = data_dictionary[key][k]['label']
		if current_label == None:
			continue
		inputs.append(current_input)
		labels.append(current_label)

	inputs = np.asarray(inputs)
	labels = np.asarray(labels)

	np.save('inputs.npy', inputs, allow_pickle=True)
	np.save('labels.npy', labels, allow_pickle=True)
	return inputs, labels


def augment_images(images,labels,num_augments):
	augmented_images = []
	augmented_labels = []
	for image,label in zip(images,labels):
		augmented_images.append(image)
		augmented_labels.append(label)

		aug_image_array = image[...,np.newaxis]

		augment_base = image.reshape((1,)+aug_image_array.shape)
		augment(augment_base,label,augmented_images,augmented_labels,num_augments)
	
	augmented_images = np.asarray(augmented_images)
	augmented_labels = np.asarray(augmented_labels)

	np.save("augmented_inputs.npy",augmented_images,allow_pickle=True)
	np.save("augmented_labels.npy",augmented_labels,allow_pickle=True)


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
	get_data('./cohn-kanade-images/', './Emotion')


if __name__ == '__main__':
	main()
