import os
# import imageio

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

	# Read in images.
	for subdir, dirs, files in os.walk(images_path):
		for file in files:
			file_path = os.path.join(subdir, file)
			key = subdir.replace(images_path, '')
			# image = imageio.imread(file_path)
			# data_dictionary[key] = [image]
			data_dictionary[key] = [1]

	# Read in numbers from labels.
	for subdir, dirs, files in os.walk(labels_path):
		for file in files:
			file_path = os.path.join(subdir, file)
			key = subdir.replace(labels_path, '')[1:] # remove first / to have same key as images
			with open(file_path, 'r') as f:
				# Remove new line character at end and cast as float
				label = float(f.read().rstrip('\n'))
				if key in data_dictionary:
					data_dictionary[key].append(label)

	data_dictionary_clean = {}

	# Remove images and labels with no label.
	for key in data_dictionary:
		if key != '' and len(data_dictionary[key]) == 2:
			data_dictionary_clean[key] = data_dictionary[key]

	return data_dictionary_clean

print(get_data('/home/awei6/course/cohn-kanade-images/', '/home/awei6/course/Emotion'))