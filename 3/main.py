from time import sleep

import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

NEVU  = 0
MELANOMA = 1

def display_images(images): 
    height, width,  channels = (28, 28, 3)
    
    for i, image in enumerate(images):
        image =  images[i].reshape(height, width, channels)
        fig, ax = plt.subplots()
        ax.clear()
        ax.imshow(image)
        ax.axis('off')
        plt.show()        
        plt.close('all')

def main(): 
    x_train_set = np.load("input_files/Xtrain_Classification1.npy")
    height, width,  channels = (28, 28, 3)
    
    print(f"x_train_set.shape {np.shape(x_train_set)}")
    y_train_set = np.load("input_files/ytrain_Classification1.npy")
    
    display_images(x_train_set)
    
if __name__ == '__main__': 
    main()