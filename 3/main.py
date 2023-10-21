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
from sklearn.utils import class_weight
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers


from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from scipy.ndimage import rotate
import cv2

# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1
HEIGHT = 28 
WIDTH = 28
CHANNELS = 3

class DataManager:
    def __init__(self, x_train, y_train):
        self.original_x_train = x_train
        self.original_y_train = y_train
        
        print(f"shape  : {x_train.shape} {y_train.shape}")
        
        ## create train and test data set2
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train, test_size=0.25 , random_state=42)
        
    def balance_data(self):
        num_nevu_images = np.sum(self.y_train == NEVU)
        num_melanoma_images = np.sum(self.y_train == MELANOMA)
        
        if num_nevu_images < num_melanoma_images : min_class = NEVU 
        else : min_class =  MELANOMA
        
        angle = [90, 180, 270]
        
        min_class_augmented = []
        min_class_augmented_labels = []
        
        npy_min_class_augmented = []
        npy_min_class_augmented_labels = []
        
        print(f"shape  : {self.x_train.shape} {self.y_train.shape}")
        
        for idx in np.where(self.not_reshape_y_train == min_class)[0]:
            image = self.x_train[idx]
            label = self.not_reshape_y_train[idx]
            
            image.reshape(HEIGHT, WIDTH, CHANNELS)
            
            for angle in np.arange(1, 4) * 90: 
                min_class_augmented.append(rotate(image, angle, reshape=False))
                min_class_augmented_labels.append(label)
            
            # Flip in X and Y axisx
            min_class_augmented.append (cv2.flip (image, 0))
            min_class_augmented_labels.append(label)
            
            min_class_augmented.append(cv2.flip (image, 1))
            min_class_augmented_labels.append(label)
        
        for idx, image in enumerate(min_class_augmented):
            npy_min_class_augmented.append(image.reshape(-1)) 
        
        npy_min_class_augmented = np.array(npy_min_class_augmented)
        npy_min_class_augmented_labels = np.array(min_class_augmented_labels)
        
        
        self.x_train = np.concatenate((self.x_train, npy_min_class_augmented.reshape(-1, HEIGHT, WIDTH, CHANNELS)), axis=0)
        self.y_train = np.concatenate((self.not_reshape_y_train, npy_min_class_augmented_labels), axis=0)
        
        ## Shuffle training data 
        permutation = np.random.permutation(len(self.x_train))
        self.x_train = self.x_train[permutation]
        self.y_train = self.not_reshape_y_train[permutation]

    def augment_via_color(self, code):
        augmented_data = []
        augmented_data_labels = []
        
        x_train =  self.x_train
        y_train =  self.not_reshape_y_train
    
        for i in range(len(x_train)):
            augmented_data.append(cv2.cvtColor(x_train[i], code))
            augmented_data_labels.append(y_train[i])
            
        augmented_data = np.array(augmented_data)
        augmented_data_labels = np.array(augmented_data_labels)
        

        
        x_train = np.concatenate((x_train, augmented_data.reshape(-1, HEIGHT, WIDTH, CHANNELS)), axis=0)
        y_train = np.concatenate((y_train, augmented_data_labels), axis=0)
        
        ## Shuffle training data
        permutation = np.random.permutation(len(x_train))
        self.x_train = x_train[permutation]
        self.y_train = y_train[permutation]

    @property
    def x_train(self):
        return self._x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS)
    
    @property 
    def not_reshape_y_train(self):
        return self._y_train
    
    @x_train.setter
    def x_train(self, x_train):
        self._x_train = x_train
    
    @property
    def x_test(self):
        return self._x_test.reshape(-1, HEIGHT, WIDTH, CHANNELS)
    
    @x_test.setter
    def x_test(self, x_test):
        self._x_test = x_test
    
    @property
    def y_train(self):
        return to_categorical(self._y_train)
    
    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train
    
    @property
    def not_reshape_y_test(self):
        return self._y_test
    
    @property
    def y_test(self):
        return to_categorical(self._y_test)
    
    @y_test.setter
    def y_test(self, y_test):
        self._y_test = y_test
        
    
class CNN:
    def __init__(self): 
        

        _model = tf.keras.models.Sequential()

        #Data augmentation - + robusto

        _model.add(tf.keras.layers.RandomFlip("horizontal_and_vertical"))  
        _model.add(tf.keras.layers.RandomRotation(0.5))
        #model.add(tf.keras.layers.RandomZoom(0.5))
        #model.add(tf.keras.layers.RandomContrast(0.5))

        # 1st convolutional layer

        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(30,30,3)))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3)) 
        
        # 2nd convolutional layer

        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3))

        
        ## 3rd convolutional layer
        #model.add(tf.keras.layers.Conv2D(32, kernel_size=2, activation='relu'))
        #model.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))
        
        # Fully connected classifier

        _model.add(tf.keras.layers.Flatten())
        

        _model.add(tf.keras.layers.Dense( 1024, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    


        _model.add(tf.keras.layers.Dense(256, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    
    #

        _model.add(tf.keras.layers.Dense(56, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    
        

        _model.add(tf.keras.layers.Dense(2, activation ='softmax'))
        #model.summary()


        _model.compile(tf.keras.optimizers.Adam(0.0002),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # loss='categorical_crossentropy',
                    
                    )    
        
        self.model = _model
            
    

def main():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    batch_size = 256
    
    professor_x_train_set = np.load("input_files/Xtrain_Classification1.npy")
    professor_y_train_set = np.load("input_files/ytrain_Classification1.npy")
    
    data_manager = DataManager(professor_x_train_set, professor_y_train_set)

    data_manager.balance_data()
    data_manager.augment_via_color(cv2.COLOR_BGR2HSV)

    print(f"x_tr : {np.shape(data_manager.x_train)} y_tr {np.shape(data_manager.y_train)} x_te {np.shape(data_manager.x_test)} y_te {np.shape(data_manager.y_test)}")
    

    train_dataset = tf.data.Dataset.from_tensor_slices((data_manager.x_train, data_manager.y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    val_dataset =  tf.data.Dataset.from_tensor_slices((data_manager.x_test, data_manager.y_test))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)

    train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    

    print(f"Building CCN")
    cnn = CNN()

    print(f"declaring callback ")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping


    print(f"declaring history")
    history = cnn.model.fit(data_manager.x_train, data_manager.y_train, epochs = 500, validation_data =train_dataset, callbacks=[callback])

    print(f"predicting")
    predict = np.argmax(cnn.model.predict(data_manager.x_test), axis=-1)

    print(f"calculating f1 score")
    results2 = metrics.f1_score(np.argmax((data_manager.y_test),  axis = -1),  predict)

    print('F1-Score', ':',results2)

    cnn.model.save('model.keras')

if __name__ == '__main__':
    main()
