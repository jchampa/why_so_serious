import os
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
			# print(key)
			num = int(file_path[-6:-4])
			if key in intense_num:
				if intense_num[key] < num:
					intense_num[key] = num
			else:
				intense_num[key] = num

			if key not in data_dictionary:
				data_dictionary[key] = {}

			image = img_to_array(load_img(file_path,color_mode="grayscale",target_size=(480,640),interpolation="nearest"))
			data_dictionary[key][num] = {'image': image, 'label': None}
			# data_dictionary[key] = [1]

	# Read in numbers from labels.
	

	counter = 0
	for subdir, dirs, files in os.walk(labels_path):
		for file in files:

			if file == '.DS_Store':
				continue

			file_path = os.path.join(subdir, file)
			# print(file_path)
			num = int(file_path[-15:-12])
			key = subdir.replace(labels_path, '')[1:] # remove first / to have same key as images
			with open(file_path, 'r') as f:
				# Remove new line character at end and cast as float
				label = float(f.read().rstrip('\n'))
				if key in data_dictionary:
					for k in data_dictionary[key]:
						#first two images have neutral label
						if k == 1 or k == 2:
							data_dictionary[key][k]['label'] = 0
						else:
							data_dictionary[key][k]['label'] = label

	inputs = []
	labels = []
	
	for key in data_dictionary:
		#this is if we only want to do first and most intense image in the folder
		for j in range(1, 5):
			if j == 3:
				k = intense_num[key]
			elif j == 4:
				k = intense_num[key] - 1
			else:
				k = j
			current_input = data_dictionary[key][k]['image']
			current_label = data_dictionary[key][k]['label']
			if current_label == None:
				continue
			inputs.append(current_input)
			labels.append(current_label)
			print(len(inputs), len(labels))
	
	inputs = np.asarray(inputs)
	labels = np.asarray(labels)

	np.save('inputs2.npy', inputs, allow_pickle=True)
	np.save('labels2.npy', labels, allow_pickle=True)
	return inputs, labels


def load_data(image_path, label_path):
	inputs = np.load(image_path, allow_pickle=True)
	labels = np.load(label_path, allow_pickle=True)
	return inputs, labels

def main():
	get_data('./cohn-kanade-images/', './Emotion')


if __name__ == '__main__':
	main()
