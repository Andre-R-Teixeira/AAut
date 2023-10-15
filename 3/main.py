# fastai and torch
#import fastai
#from fastai.metrics import accuracy
#from fastai.vision import (
#    models, ImageList, imagenet_stats, partial, cnn_learner, ClassificationInterpretation, to_np,
#)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from keras_tuner import RandomSearch, Hyperband
import tensorflow_addons as tfa
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
import keras_tuner as kt
from imblearn.over_sampling import SMOTE


# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1


class CNN:
    def __init__(self): 
        
        print(f"0")
        _model = tf.keras.models.Sequential()

        #Data augmentation - + robusto
        print(f"1");
        _model.add(tf.keras.layers.RandomFlip("horizontal_and_vertical"))  
        _model.add(tf.keras.layers.RandomRotation(0.5))
        #model.add(tf.keras.layers.RandomZoom(0.5))
        #model.add(tf.keras.layers.RandomContrast(0.5))

        # 1st convolutional layer
        print(f"2")
        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(30,30,3)))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3)) 
        
        # 2nd convolutional layer
        
        print(f"3")
        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3))

        
        ## 3rd convolutional layer
        #model.add(tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu'))
        #model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
        
        # Fully connected classifier
        print(f"4")
        _model.add(tf.keras.layers.Flatten())
        
        # 1st dense layer
        print(f"5")
        _model.add(tf.keras.layers.Dense( 1024, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    

        # 2st dense layer
        print(f"6")
        _model.add(tf.keras.layers.Dense(256, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    
    #
        ## 3st dense layer
        print(f"7")
        _model.add(tf.keras.layers.Dense(56, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    
        
        # Output layer
        print(f"8")
        _model.add(tf.keras.layers.Dense(2, activation ='softmax'))
        #model.summary()

        # Compile~
        print(f"9")
        _model.compile(tf.keras.optimizers.Adam(0.0002),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # loss='categorical_crossentropy',
                    metrics=['accuracy', tfa.metrics.F1Score(num_classes=2)])    
        
        self.model = _model
            
        

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
    #1gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    #1for device in gpu_devices:
    #1    tf.config.experimental.set_memory_growth(device, True)
    
    x_train_set = np.load("image_rotated.npy")
    y_train_set = np.load("image_classification.npy")
    
    ## Separate the data into nevus and melanomas    
    melanoma_x_train_set = x_train_set[y_train_set == MELANOMA]
    melanoma_y_train_set = np.ones(np.shape(melanoma_x_train_set)[0])
    
    nevu_x_train_set = x_train_set[y_train_set == NEVU]
    nevu_y_train_set = np.zeros(np.shape(nevu_x_train_set)[0]) 
    
    np.save('melanoma_x_train_set.npy', melanoma_x_train_set)
    np.save('melanoma_y_train_set.npy', melanoma_y_train_set)
    
    np.save('nevu_x_train_set.npy', nevu_x_train_set)
    np.save('nevu_y_train_set.npy', nevu_y_train_set)
    
    ## combine and shuffle data 
    X  = np.concatenate((melanoma_x_train_set, nevu_x_train_set), axis = 0)
    y = np.concatenate((np.ones(melanoma_x_train_set.shape[0]), np.zeros(nevu_x_train_set.shape[0])), axis = 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Building CCN")
    cnn = CNN()

    print(f"declaring callback ")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping
    
    print(f"declaring history")
    history = cnn.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), callbacks=[callback])


    print(f"predicting")
    predict = np.argmax(cnn.model.predict(X_test), axis=-1)
    
    
    print(f"calculating f1 score")
    results2 = metrics.f1_score(np.argmax((y_test),  axis = -1),  predict)
    
    
    print('F1-Score', ':',results2)


if __name__ == '__main__':
    main()
