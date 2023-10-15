import numpy as np

#import tensorflow as tf

import matplotlib.pyplot as plt

from scipy.ndimage import rotate


# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1

def rotate_images_to_balance(images, classification):
    num_nevu_images = np.sum(classification == NEVU)
    num_melanoma_images = np.sum(classification == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}")


    height, width,  channels = (28, 28, 3)
    angles = [45, 90, 180, 270, 25]

    rotated_images_list = []
    npy_images_list = []
    classification_array = []

    if num_nevu_images < num_melanoma_images:
        class_to_rotate = NEVU
    else:
        class_to_rotate = MELANOMA

    indices_to_rotate = np.where(classification == class_to_rotate)[0]

    rotated_images = []
    for idx in indices_to_rotate:
        image = images[idx]
        image =  images[idx].reshape(height, width, channels)

        for ang in angles:
            rotated_images_list.append(rotate(image, ang, reshape=False))
            classification_array.append(classification[idx])

    for idx, image in enumerate(rotated_images_list):
        npy_images_list.append(image.reshape(-1)) 


    rotated_images = np.array(npy_images_list)
    new_labels = np.array(classification_array)

    # Stack the rotated images with the original dataset
    images_list = np.vstack((images, rotated_images))
    classi = np.concatenate((classification, new_labels))



    np.save('image_classification.npy', np.array(classi))
    np.save('image_rotated.npy', np.array(images_list))

def main():
    x_train_set = np.load("input_files/Xtrain_Classification1.npy")
    y_train_set = np.load("input_files/ytrain_Classification1.npy")
    
    print(f"x_train_set.shape {np.shape(x_train_set)}")
    
    # Rotate images to balance the dataset
    rotate_images_to_balance(x_train_set, y_train_set)

    clas = np.load('image_classification.npy')
    image = np.load('image_rotated.npy')
    
    print(f"classi shape : {np.shape(clas)}  image shape : {np.shape(image)}")

    num_nevu_images = np.sum(clas == NEVU)
    num_melanoma_images = np.sum(clas == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}")

   

if __name__ == '__main__':
    main()





   

