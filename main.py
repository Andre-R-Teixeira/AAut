import numpy as np
import cv2
from scipy.ndimage import rotate


# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1

def rotate_images_to_balance(images,classification):
    num_nevu_images = np.sum(classification == NEVU)
    num_melanoma_images = np.sum(classification == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}")

    if num_nevu_images < num_melanoma_images:
        class_to_rotate = NEVU
    else:
        class_to_rotate = MELANOMA

    indices_rotation = np.where(classification == class_to_rotate)[0]

    angle = [25, 45, 90, 180, 270]

    rotation(images,classification,indices_rotation,angle)

    

def rotation(images,classification,indices_to_rotate,angles):

    height, width,  channels = (28, 28, 3)
    rotated_images_list = []
    npy_images_list = []
    classification_array = []

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

def adjust_brightness_contrast(images, classification,index,alpha, beta):

    adjusted_img_list=[]
    adjusted_img = []
    classification_array=[]
    height, width,  channels = (28, 28, 3)
    
    for i in index:
        image = images[i]
        image =  images[i].reshape(height, width, channels)

        aux=cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        adjusted_img_list.append(aux)
        classification_array.append(classification[i])
       

    adjusted_img = np.array(adjusted_img_list)
    new_labels = np.array(classification)

   
    images_list = np.vstack((images, adjusted_img.reshape(-1, images.shape[1])))
    classi = np.concatenate((classification, new_labels))



    np.save('image_classification.npy', np.array(classi))
    np.save('image_bright.npy', np.array(images_list))

    

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

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}\n\n")

    id = range(len(clas))
    degree = [101,73,300]

    rotation(image,clas,id,degree)

    class_t = np.load('image_classification.npy')
    image_t = np.load('image_rotated.npy')

    print(f"classi shape : {np.shape(class_t)}  image shape : {np.shape(image_t)}")

    num_nevu_images = np.sum(class_t == NEVU)
    num_melanoma_images = np.sum(class_t == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}\n\n")

    ids = range(len(class_t))
    alpha = 0.5
    beta = -50

    adjust_brightness_contrast(image_t,class_t,ids,alpha,beta)


    class_all = np.load('image_classification.npy')
    image_all = np.load('image_bright.npy')

    print(f"classi shape : {np.shape(class_all)}  image shape : {np.shape(image_all)}")

    num_nevu_images = np.sum(class_all == NEVU)
    num_melanoma_images = np.sum(class_all == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}")
   

if __name__ == '__main__':
    main()








   

