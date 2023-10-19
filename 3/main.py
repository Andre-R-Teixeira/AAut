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
                    
                    )    
        
        self.model = _model
            
    

def main():
    batch_size = 256
    
    x_train_set = np.load("image_colored.npy")
    y_train_set = np.load("image_classification.npy")

    ## Separate the data into nevus and melanomas    
    melanoma_x_train_set = x_train_set[y_train_set == MELANOMA]
    melanoma_y_train_set = np.ones(np.shape(melanoma_x_train_set)[0])

    nevo_x_train_set = x_train_set[y_train_set == NEVU]
    nevo_y_train_set = np.zeros(np.shape(nevo_x_train_set)[0]) 

    np.save('melanoma_x_train_set.npy', melanoma_x_train_set)
    np.save('melanoma_y_train_set.npy', melanoma_y_train_set)
    np.save('nevo_x_train_set.npy', nevo_x_train_set)
    np.save('nevo_y_train_set.npy', nevo_y_train_set)

    ## combine and shuffle data 
    X  = np.concatenate((melanoma_x_train_set, nevo_x_train_set), axis = 0)
    y = np.concatenate((np.ones(melanoma_x_train_set.shape[0]), np.zeros(nevo_x_train_set.shape[0])), axis = 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

    X_train = X_train.reshape(-1,28,28,3)
    X_test = X_test.reshape(-1,28,28,3)

    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)
    
    
 

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    val_dataset =  tf.data.Dataset.from_tensor_slices((X_test, y_test))
    val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
    
    train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    print(f"Building CCN")
    cnn = CNN()

    print(f"declaring callback ")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping


    print(f"declaring history")
    history = cnn.model.fit(X_train, y_train, epochs = 200, validation_data =train_dataset, callbacks=[callback])

    print(f"predicting")
    predict = np.argmax(cnn.model.predict(X_test), axis=-1)

    print(f"calculating f1 score")
    results2 = metrics.f1_score(np.argmax((y_test),  axis = -1),  predict)

    print('F1-Score', ':',results2)

    cnn.model.save('model.keras')

if __name__ == '__main__':
    main()
