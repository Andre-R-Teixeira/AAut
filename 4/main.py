# fastai and torch
#import fastai
#from fastai.metrics import accuracy
#from fastai.vision import (
#    models, ImageList, imagenet_stats, partial, cnn_learner, ClassificationInterpretation, to_np,
#)

import numpy as np

from collections import Counter


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


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

from scipy.ndimage import rotate
import cv2

# Assuming NEVU is class 0 and MELANOMA is class 1
NEVU = 0
MELANOMA = 1
VASCULAR_LESIONS = 2
GRANULOCYTES = 3
BASOPHILS = 4
LYMPHOCYTES = 5

DERMOSCOPIC_IMAGES = 0
BLOOD_CELL_MICROSCOPY = 1 

HEIGHT = 28 
WIDTH = 28
CHANNELS = 3

class DecisionTree:
    def __init__(self,min_samples_split=2): 
        self.clf = DecisionTreeClassifier(min_samples_split=min_samples_split)
        
        
    def train(self, x_train, y_train): 
        self.clf = self.clf.fit(x_train.reshape(x_train.shape[0], -1), y_train)
    
    def predict (self, x_test): 
        return self.clf.predict(x_test.reshape(x_test.shape[0], -1))

class DataManager:
    def __init__(self, x_train, y_train):
        self.original_x_train = x_train
        self.original_y_train = y_train
        
        print(f"shape  : {x_train.shape} {y_train.shape}")

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x_train, y_train, test_size=0.20 , random_state=42)

    def balance_training_set(self ):
        data_counter = Counter(self.y_train)

        major_class = data_counter.most_common(1)[0]
        
        most_common_value = major_class[0]
        most_common_count = major_class[1]

        # Create an ImageDataGenerator for data augmentation

        ## list with 90, 180, 270 degree for rotation of image
        angle_rotation = np.arange(1, 4) * 90;
        
        full_data_set = []
        full_classified_set = []

        full_data_set =  np.array(full_data_set).reshape(-1, HEIGHT * WIDTH * CHANNELS)

        ## Get all older images
        for i in np.arange(0, 6):
            print(f"np.shape {np.shape(np.array(full_data_set))}")
            print(f"np.shape {np.shape(self.class_images(i))}")

            full_data_set =  np.concatenate((np.array(full_data_set), self.class_images(i) ))
            full_classified_set =  np.concatenate((np.array(full_classified_set), np.full(np.shape(self.class_images(i))[0],  i)))

        array = np.arange(0, 6)
        array =  array[array != most_common_value]
        
        ## Augment data except for the most common class
        for i in array:
            
            current_class_images = self.class_images(i)
            num_images_current_class = np.shape(current_class_images)[0]

            # Generate augmented samples for the minority class
            augmented_samples = []
            classified_samples = []

            for angle in angle_rotation:
                for image in current_class_images:
                    augmented_samples.append(rotate(image.reshape(HEIGHT, WIDTH, CHANNELS), angle, reshape=False))
                    classified_samples.append(i)
            
                

            # Combine the augmented samples with the original dataset
            full_data_set = np.concatenate((np.array(full_data_set), np.array(augmented_samples).reshape(-1, HEIGHT * WIDTH * CHANNELS))) 
            full_classified_set = np.concatenate((np.array(full_classified_set), classified_samples))

        self.x_train = full_data_set
        self.y_train = full_classified_set

    def class_images (self, class_type): 
        return self.x_train[np.where(self.y_train == class_type)]
    
    def dataset_images (self, dataset_type = 0):
        if dataset_type == DERMOSCOPIC_IMAGES:
            return [self.x_train[np.where(self.y_train < 3)], 
                    self.y_train[np.where(self.y_train < 3)]]
            
        elif dataset_type == BLOOD_CELL_MICROSCOPY:
            return [self.x_train[np.where(self.y_train > 3)],
                    self.y_train[np.where(self.y_train > 3)]]
    
    def save_train_data (self, x_train_name, x_class_name ): 
        print(f"x_train_shape {np.shape(self.x_train)}")
        np.save(x_train_name, self.x_train)
        print(f"y_train_shape {np.shape(self.y_train)}")
        np.save(x_class_name, self.y_train)
    
    @property
    def x_train(self):
        return self._x_train
    
    @x_train.setter
    def x_train(self, x_train):
        self._x_train = x_train
    
    @property
    def x_test(self):
        return self._x_test
    
    @x_test.setter
    def x_test(self, x_test):
        self._x_test = x_test
    
    @property
    def y_train(self):
        return self._y_train
    
    @y_train.setter
    def y_train(self, y_train):
        self._y_train = y_train
    
    @property
    def y_test(self):
        return self._y_test
    
    @y_test.setter
    def y_test(self, y_test):
        self._y_test = y_test
        
    
class CNN:
    def __init__(self, name = "cnn"): 
        self.name  = name
        

        _model = tf.keras.models.Sequential()

        #Data augmentation - + robusto

        _model.add(tf.keras.layers.RandomFlip("horizontal_and_vertical"))  
        _model.add(tf.keras.layers.RandomRotation(0.5))
        #model.add(tf.keras.layers.RandomZoom(0.5))
        #model.add(tf.keras.layers.RandomContrast(0.5))

        # 1st convolutional layer

        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,3)))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3)) 
        
        # 2nd convolutional layer

        _model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
        _model.add(tf.keras.layers.MaxPool2D(pool_size=3,strides=3))

        # Fully connected classifier

        _model.add(tf.keras.layers.Flatten())

        _model.add(tf.keras.layers.Dense( 1024, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    


        _model.add(tf.keras.layers.Dense(256, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    

        _model.add(tf.keras.layers.Dense(56, activation='relu'))
        _model.add(tf.keras.layers.Dropout(0.5))    
        

        _model.add(tf.keras.layers.Dense(3, activation ='softmax'))

        _model.compile( tf.keras.optimizers.Adam(0.0002),
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'] )
        # custom balance accuracy    

        self.model = _model

    def save(self): 
        self.model.save(self.name)
        
def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    batch_size = 256

    professor_x_train_set = np.load("input_files/Xtrain_Classification2.npy")
    professor_y_train_set = np.load("input_files/ytrain_Classification2.npy")

    data_manager =  DataManager(professor_x_train_set, professor_y_train_set)
    data_manager.balance_training_set()

    data_manager.save_train_data("augmented_x_train.npy", "augmented_y_train.npy")

    x_train  = data_manager.x_train
    x_test = data_manager.x_test

    y_train = data_manager.y_train
    y_test = data_manager.y_test

    y_train_classifier =  np.where(y_train < 3, 0, 1)
    y_test_classifier =  np.where(y_test < 3, 0, 1)

    print(f"trainig   : {x_train.shape} {y_train.shape}")
    print(f"testing: {x_test.shape} {y_test.shape}")

    clf = DecisionTree(min_samples_split = 2)
    clf.train(data_manager.x_train, y_train_classifier)
    y_pred = clf.predict(x_test)

    ## get training set images classified as dermoscopic and blood cell microscopy
    dermoscopy_x_train = x_train[np.where(y_train < 3)]  
    dermoscopy_y_train = y_train[np.where(y_train < 3)]
    dermoscopy_x_test = x_test[np.where(y_test < 3)]
    dermoscopy_y_test = y_test[np.where(y_test < 3)]


    blood_cell_x_train = x_train[np.where(y_train > 3)]
    blood_cell_y_train  = y_train[np.where(y_train > 3)]
    blood_cell_y_train = blood_cell_y_train - 3
    blood_cell_x_test = x_test[np.where(y_test > 3)]
    blood_cell_y_test = y_test[np.where(y_test > 3)]
    blood_cell_y_test = blood_cell_y_test - 3

    print(f"dermoscopy_images {np.shape(dermoscopy_x_train)} ,dermoscopy_classified {np.shape(dermoscopy_y_train)}")
    print(f"blood_cell_images {np.shape(blood_cell_x_train)} ,blood_cell_classified {np.shape(blood_cell_y_train)}")

    dermoscopy_cnn = CNN("dermoscopy_cnn")

    blood_cell_cnn = CNN("blood_cell_cnn")

    dermoscopy_train  = tf.data.Dataset.from_tensor_slices((dermoscopy_x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS), dermoscopy_y_train))
    dermoscopy_train = dermoscopy_train.shuffle(buffer_size=1024).batch(batch_size)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping    
    history = dermoscopy_cnn.model.fit(dermoscopy_x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS), dermoscopy_y_train, epochs = 500, validation_data =dermoscopy_train, callbacks=[callback])

    blood_cell_train  = tf.data.Dataset.from_tensor_slices((blood_cell_x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS), blood_cell_y_train))
    blood_cell_train = blood_cell_train.shuffle(buffer_size=1024).batch(batch_size)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True) # early stopping    
    history = blood_cell_cnn.model.fit(blood_cell_x_train.reshape(-1, HEIGHT, WIDTH, CHANNELS), blood_cell_y_train, epochs = 500, validation_data =blood_cell_train, callbacks=[callback])

    print(f"predictions")
    print(f"Decision tree arcuracy : {metrics.accuracy_score(y_test_classifier, y_pred)}")
    dermoscopy_predict = np.argmax(dermoscopy_cnn.model.predict(dermoscopy_x_test.reshape(-1, HEIGHT, WIDTH, CHANNELS)), axis=-1)
    blood_predict = np.argmax(blood_cell_cnn.model.predict(blood_cell_x_test.reshape(-1, HEIGHT, WIDTH, CHANNELS)), axis=-1)


    print(f"dermoscopy_predict : {dermoscopy_predict}")

    dermoscopy_confusion_matrix = confusion_matrix (dermoscopy_predict, dermoscopy_y_test)
    blood_cell_confusion_matrix = confusion_matrix (blood_predict, blood_cell_y_test)
    
    dermoscopy_balanced_accuracy = balanced_accuracy_score(dermoscopy_y_test, dermoscopy_predict)
    blood_cell_balanced_accuracy = balanced_accuracy_score(blood_cell_y_test, blood_predict)
    
    print(f"dermoscopy_confusion_matrix {dermoscopy_confusion_matrix}")
    print(f"blood_cell_confusion_matrix {blood_cell_confusion_matrix}")
    
    print(f"dermoscopy_balanced_accuracy {dermoscopy_balanced_accuracy}")
    print(f"blood_cell_balanced_accuracy {blood_cell_balanced_accuracy}")
    
    dermoscopy_cnn.save()
    blood_cell_cnn.save()
    
#    print(f"predicting")
#    predict = np.argmax(cnn.model.predict(x_test), axis=-1)
#
#    print(f"calculating f1 score")
#    results2 = metrics.f1_score(y_test, predict, average='micro')
#
#    print('F1-Score', ':',results2)
#
#    cnn.model.save('model.keras')

if __name__ == '__main__':
    main()
