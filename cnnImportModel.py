from __future__ import absolute_import
import math
import os
import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Reshape
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split



def main():
	
	labelDict = {0: 'neutral', 1: 'anger', 2: 'contempt', 3: 'disgust', 4:'fear', 5: 'happy', 6: 'sadness', 7: 'surprise'} 
	# 0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise

	
	new_model = tf.keras.models.load_model('vanillaModel.h5')

	file_path = 'smiley.jpg'
	
	img = load_img(file_path,color_mode="grayscale",target_size=(480,640),interpolation="nearest")
	image = img_to_array(img)
	image = image[np.newaxis, ...]
	image = (np.asarray(image)/255.0).astype(np.float32)


	prediction = new_model.predict(image)
	print("probabilities: ", prediction[0])
	pred = np.argmax(prediction[0])
	print("class label: ", pred)
	print("Emotion: ", labelDict[pred])


if __name__ == '__main__':
	main()
