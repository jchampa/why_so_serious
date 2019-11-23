import os
import imageio
import numpy as np
import cv2
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
			# print(file_path)
			# print(subdir)
			key = subdir.replace(images_path, '')
			# print(key)
			num = int(file_path[-6:-4])
			if key in intense_num:
				if intense_num[key] < num:
					intense_num[key] = num
			else:
				intense_num[key] = num
			image = cv2.resize(cv2.imread(file_path, 0), (480, 640))
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
	print('rgbs: ')
	print(rgb)

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
						if k == 1:
							data_dictionary[key][k]['label'] = 0
						else:
							data_dictionary[key][k]['label'] = label

	inputs = []
	labels = []
	# for key in data_dictionary:
	# 	print(key)
	# 	for k in data_dictionary[key]:
	# 		print(k)
	# 		print(data_dictionary[key][k]['image'].shape)
	# 		print(data_dictionary[key][k]['label'])
	# 	print('')
	# Remove images and labels with no label.
	checker = []
	for key in data_dictionary:
		#this is if we only want to do first and most intense image in the folder
		# k = intense_num[key]
		# current_input = data_dictionary[key][k]['image']
		# current_label = data_dictionary[key][k]['label']
		# if label == None:
		# 	continue
		# inputs.append(current_input)
		# labels.append(current_label)
		for k in data_dictionary[key]:

			current_input = data_dictionary[key][k]['image']
			current_label = data_dictionary[key][k]['label']
			if current_label:
				checker.append((key, k))
				inputs.append(current_input)
				labels.append(current_label)

		# if key != '' and len(data_dictionary[key]) == 2:
		# 	current_input = data_dictionary[key][0]
		# 	current_label = data_dictionary[key][1]
		# 	inputs.append(current_input)
		# 	labels.append(current_label)

	#just testing to make sure it works
	# for i in range(len(inputs)):
	# 	print(inputs[i])
	# 	print(labels[i])
	# 	print(checker[i])
	# 	print('\n\n\n')
	inputs = np.asarray(inputs)
	labels = np.asarray(labels)

	np.save('inputs.npy', inputs)
	np.save('labels.npy', labels)
	return inputs, labels

get_data('./cohn-kanade-images/', './Emotion')

def load_data(image_path, label_path):
	inputs = np.load(image_path)
	labels = np.load(label_path)
	return inputs, labels
