import numpy as np
import cv2
from scipy.ndimage import rotate
import random
from sklearn.model_selection import train_test_split



# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1
VASCULAR_LESIONS = 2
GRANULOCYTES = 3
BASOPHILS = 4
LYMPHOCYTES = 5


def flip_to_balance(images,classification,indices):
    height, width,  channels = (28, 28, 3)
    rotated_images_list = []
    npy_images_list = []
    classification_array = []

    rotated_images = []
    for idx in indices:
        image = images[idx]
        image =  images[idx].reshape(height, width, channels)


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
    np.save('rotated_images', np.array(images_list))
    

    

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
    np.save('rotated_images', np.array(images_list))


def adjust_brightness_contrast(images, classification,index):

    adjusted_img_list=[]
    adjusted_img = []
    classification_array=[]
    height, width,  channels = (28, 28, 3)
    
    for i in index:
        image = images[i]
        image =  images[i].reshape(height, width, channels)

        alpha = np.random.normal(1.0,0.1)
        beta = np.random.normal(0, 0.1)

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
    

    for i in index:
        image = images[i]
        image = images[i].reshape(height, width, channels)

        
        color_image = cv2.cvtColor(image, code)

        color_img_list.append(color_image)
        classification_array.append(classification[i])

    colored_img = np.array(color_img_list)
    new_labels = np.array(classification_array)

    images_list = np.vstack((images, colored_img.reshape(-1, images.shape[1])))
    classi = np.concatenate((classification, new_labels))

    np.save('image_classification.npy', np.array(classi))
    np.save('image_colored.npy', np.array(images_list))





def main():
    x_train_set = np.load("input_files/Xtrain_Classification2.npy")
    y_train_set = np.load("input_files/ytrain_Classification2.npy")

    X_train, X_test, y_train, y_test = train_test_split(x_train_set, y_train_set, test_size=0.2, random_state=42)


    np.save('ytest.npy', np.array(y_test))
    np.save('xtest.npy', np.array(X_test))

    np.save('image_classification.npy', np.array(y_train))
    np.save('image.npy', np.array(X_train))

    img= np.load('image.npy')
    cla= np.load('image_classification.npy')


    print(f"x_train_set.shape {np.shape(img)}")
    
    
    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")

    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)

    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")


    class_to_rotate = MELANOMA

    indices_rotation = np.where(cla == class_to_rotate)[0]
    dg=[90,180,270]

    rotation(img,cla,indices_rotation,dg)

    img= np.load('rotated_images.npy')
    cla= np.load('rotated_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")

    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)


    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")

    class_to_rotate = GRANULOCYTES

    indices_rotation = np.where(cla == class_to_rotate)[0]
    

    flip_to_balance(img,cla,indices_rotation)

    img= np.load('rotated_images.npy')
    cla= np.load('rotated_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")

    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)



    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")


    class_to_rotate = BASOPHILS

    indices_rotation = np.where(cla == class_to_rotate)[0]
    dg=[90,180,270]

    rotation(img,cla,indices_rotation,dg)

    img= np.load('rotated_images.npy')
    cla= np.load('rotated_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")

    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)



    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")


    class_to_rotate = LYMPHOCYTES

    indices_rotation = np.where(cla == class_to_rotate)[0]
    dg=[90,180,270]

    rotation(img,cla,indices_rotation,dg)

    img= np.load('rotated_images.npy')
    cla= np.load('rotated_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")

    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)


    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")


    class_to_rotate = VASCULAR_LESIONS

    indices_rotation = np.where(cla == class_to_rotate)[0]
    dg=[90,180,270]

    rotation(img,cla,indices_rotation,dg)

    img= np.load('rotated_images.npy')
    cla= np.load('rotated_classification.npy')

    indices_bright = np.where(cla == class_to_rotate)[0]

    adjust_brightness_contrast(img,cla,indices_bright)

    img= np.load('image_bright.npy')
    cla= np.load('image_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")


    indices_color = np.where(cla == class_to_rotate)[0]

    color(img,cla,indices_color, cv2.COLOR_BGR2Lab)

    img= np.load('image_colored.npy')
    cla= np.load('image_classification.npy')

    indices_color = np.where(cla == class_to_rotate)[0]

    color(img,cla,indices_color, cv2.COLOR_BGR2YUV)

    img= np.load('image_colored.npy')
    cla= np.load('image_classification.npy')


    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")


    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)



    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")


    indice = range(len(cla))
    
    adjust_brightness_contrast(img,cla,indice)

    img= np.load('image_bright.npy')
    cla= np.load('image_classification.npy')

    print(f"classi shape : {np.shape(cla)}  image shape : {np.shape(img)}")


    num_nevu_images = np.sum(cla == NEVU)
    num_melanoma_images = np.sum(cla == MELANOMA)
    num_vascular_lesions_images = np.sum(cla == VASCULAR_LESIONS)
    num_granulocytes_images = np.sum(cla == GRANULOCYTES)
    num_basophils_images = np.sum(cla == BASOPHILS)
    num_lymphocytes_images = np.sum(cla == LYMPHOCYTES)



    print(f"nevu : {num_nevu_images}  melanoma: {num_melanoma_images}  vascular lesions: {num_vascular_lesions_images} \n")
    print(f"granulocytes : {num_granulocytes_images}  basophils: {num_basophils_images}  lymphocypes: {num_lymphocytes_images} \n")







if __name__ == '__main__':
    main()
