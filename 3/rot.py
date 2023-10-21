import numpy as np
import cv2
from scipy.ndimage import rotate
import random



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

    angle = [90, 180, 270]

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

         # Flip the image horizontally

        rotated_images_list.append(cv2.flip(image, 1)) 
        classification_array.append(classification[idx])

         # Flip the image vertically
        rotated_images_list.append(cv2.flip(image, 0)) 
        classification_array.append(classification[idx])


    for idx, image in enumerate(rotated_images_list):
        npy_images_list.append(image.reshape(-1)) 


    rotated_images = np.array(npy_images_list)
    new_labels = np.array(classification_array)

    # Stack the rotated images with the original dataset
    images_list = np.vstack((images, rotated_images))
    classi = np.concatenate((classification, new_labels))



    np.save('rotated_classification.npy', np.array(classi))
    np.save('rotated_images.npy', np.array(images_list))

def adjust_brightness_contrast(images, classification,index):

    adjusted_img_list=[]
    adjusted_img = []
    classification_array=[]
    height, width,  channels = (28, 28, 3)
    
    for i in index:
        image = images[i]
        image =  images[i].reshape(height, width, channels)

        alpha = np.random.normal(1.0,0.2)
        beta = np.random.normal(0, 0.2)

        aux=cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        adjusted_img_list.append(aux)
        classification_array.append(classification[i])


    adjusted_img = np.array(adjusted_img_list)
    new_labels = np.array(classification_array)


    images_list = np.vstack((images, adjusted_img.reshape(-1, images.shape[1])))
    classi = np.concatenate((classification, new_labels))



    np.save('image_classification.npy', np.array(classi))
    np.save('image_bright.npy', np.array(images_list))


def color(images, classification, index, code):
    color_img_list = []
    classification_array = []
    height, width, channels = (28, 28, 3)
    
    
    print(f"images : {np.shape(images)}  classification : {np.shape(classification)}")

    j = 0

    for i in index:
        j+=1
        image = images[i]
        image = images[i].reshape(height, width, channels)

        color_image = cv2.cvtColor(image, code)


        color_img_list.append(color_image)
        classification_array.append(classification[i])
        

    colored_img = np.array(color_img_list)
    new_labels = np.array(classification_array)
    print(f"color_img_list : {np.shape(colored_img)}  new_labels : {np.shape(new_labels)}")

    images_list = np.vstack( (images, colored_img.reshape(-1, 28 * 28 * 3)))
    classi = np.concatenate( (classification, new_labels))

    print(f"images_list shape : {np.shape(images_list)}  classi shape : {np.shape(classi)}")

    np.save('image_classification.npy', np.array(classi))
    np.save('image_colored.npy', np.array(images_list))





def main():
    x_train_set = np.load("input_files/Xtrain_Classification1.npy")
    y_train_set = np.load("input_files/ytrain_Classification1.npy")
    
    print(f"x_train_set.shape {np.shape(x_train_set)}")
    
    # Rotate images to balance the dataset
    rotate_images_to_balance(x_train_set, y_train_set)


    clas = np.load('rotated_classification.npy')
    image = np.load('rotated_images.npy')
    
    print(f"classi shape : {np.shape(clas)}  image shape : {np.shape(image)}")

    num_nevu_images = np.sum(clas == NEVU)
    num_melanoma_images = np.sum(clas == MELANOMA)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}\n\n")

   

    ids = range(len(clas))
    

    adjust_brightness_contrast(image,clas,ids)


    class_all = np.load('image_classification.npy')
    image_all = np.load('image_bright.npy')


    num_nevu_images = np.sum(class_all == NEVU)
    num_melanoma_images = np.sum(class_all == MELANOMA)


    i = range(len(class_all))
    c=cv2.COLOR_BGR2Lab
    
    color(image_all,class_all,i, c)
   
    class_f = np.load('image_classification.npy')
    image_f = np.load('image_colored.npy')


    num_nevu_images = np.sum(class_f == NEVU)
    num_melanoma_images = np.sum(class_f == MELANOMA)



    #ie = range(len(class_f))
    #cl=cv2.COLOR_BGR2GRAY
    
    #color(image_f, class_f, ie, cl)
    
    #class_t = np.load('image_classification.npy')
    #image_t = np.load('image_colored.npy')

    #num_nevu_images = np.sum(class_t == NEVU)
    #num_melanoma_images = np.sum(class_t == MELANOMA)




if __name__ == '__main__':
    main()
