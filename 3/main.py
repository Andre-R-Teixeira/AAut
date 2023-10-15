import numpy as np

#import tensorflow as tf

import matplotlib.pyplot as plt

import torch 
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from scipy.ndimage import rotate

#from tensorflow.keras import datasets, layers, models

NEVU  = 0
MELANOMA = 1


def display_images(images, classification): 
    height, width,  channels = (28, 28, 3)
    angles = list(range(0, 361, 45))

    rotated_images_list = []
    npy_images_list = []
    classification_array = []

    for i,image in enumerate(images):
        image =  images[i].reshape(height, width, channels)

        for angle in angles: 
            rotated_images_list.append(rotate(image, angle, reshape=False))
            classification_array.append(classification[i])

    for i, image in enumerate(rotated_images_list):
        npy_images_list.append(image.reshape(-1)) 

    np.save('image_classification.npy', np.array(classification_array))
    np.save('image_rotated.npy', np.array(npy_images_list))



def main(): 
    x_train_set = np.load("input_files/Xtrain_Classification1.npy")
    y_train_set = np.load("input_files/ytrain_Classification1.npy")
    
    print(f"x_train_set.shape {np.shape(x_train_set)}")

    display_images(x_train_set, y_train_set)

    classi = np.load('image_classification.npy')
    image = np.load('image_rotated')
    
    print(f"classi shape : {np.shape(classi)}  image shape : {np.shape(image)}")

if __name__ == '__main__': 
    main()