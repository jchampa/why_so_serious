import os
import numpy as np
import tensorflow as tf
import time
import dlib
from tensorflow.keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img,save_img

def get_landmarks(images, labels):

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    win = dlib.image_window()

    landmarks = []
    l = labels.astype(np.int64)
    one_hot_labels = np.zeros((l.size, int(l.max()+1)))
    one_hot_labels[np.arange(l.size),l] = 1

    count = 0

    for image,label in zip(images,labels):
        
        save_img("temp.png",image)
        img = dlib.load_grayscale_image("temp.png")
        
        dets = detector(img, 1)
        face_rect = dets[0]

        landmarks.append(np.matrix([[p.x, p.y] for p in predictor(img, face_rect).parts()]))


    print(len(one_hot_labels))
    print(len(landmarks))

    np.save("landmarks.npy",landmarks)
    np.save("one_hot_labels.npy",one_hot_labels)
    
    # return landmarks,one_hot_labels

def calc_landmark_spacial(landmarks):
    landmark_spacials = []
    for landmark_group in landmarks:
        spacial_distances = []
        for landmark_1 in landmark_group:
            spacial_distances.append([])
            for landmark_2 in landmark_group:
                spacial_distances[-1].append(np.linalg.norm(landmark_1-landmark_2))
        spacial_distances = np.asmatrix(spacial_distances)
        landmark_spacials.append(spacial_distances)
    
    landmark_spacials = np.asarray(landmark_spacials)
    print(landmark_spacials.shape)
    np.save("landmark_spacial_info.npy",landmark_spacials)

    


