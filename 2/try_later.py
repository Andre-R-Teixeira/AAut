import sys 

import random 

import numpy as  np 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression 

def  try_later(): 
    x_training_set = np.load('input_files/X_train_regression2.npy')
    y_training_set = np.load('input_files/y_train_regression2.npy')
    x_test_set = np.load('input_files/X_test_regression2.npy')

    
    SSE = []
    y_pred = []
    
    gradients = []
    
    x_train_set_cpy = np.copy(x_training_set)
    y_train_set_cpy = np.copy(y_training_set)
    
    
    b_model_regression = []
    
    
    

    for i in range (len(x_training_set)):
        model = LinearRegression().fix(x_train_set_cpy, y_train_set_cpy)
        
        # model.coef_ gives us the coeficients for each of the x's
        gradients.append(model.coef_)
        
        if i is not 0:
            if (np.linalg.norm(gradients[i] - gradients[i - 1]) < 0.4 * np.linalg.norm(gradients[i - 1] - gradients[i - 2])):
                print(f"\nNumber of outliers: {i-1}\n")
                break
            
        
        x_train_fi = x_train_set_cpy
        y_train_fi = y_train_set_cpy
        y_pred = model.predict(x_train_fi)

        b_model_regression.append(np.argmax(np.abs(y_pred, y_train_fi)))
        
        x_train_set_cpy = np.delete(x_train_set_cpy, obj=b_model_regression[i], axis=0)
        y_train_set_cpy = np.delete(y_train_set_cpy, obj=b_model_regression[i], axis=0)        

    a_model_regression = [x for x in [i for i in range(100)] if x not in b_model_regression]
    
    a_x_train_set = np.delete(np.copy(x_training_set), obj=b_model_regression, axis=0)
    b_x_train_set = np.delete(np.copy(x_training_set), obj=a_model_regression, axis=0)
    
    a_y_train_set = np.delete(np.copy(y_training_set), obj=a_model_regression, axis=0)
    b_y_train_set = np.delete(np.copy(y_training_set), obj=b_model_regression, axis=0)  

    for i in range(len(a_x_train_set)):
        a_x_train_set_cpy = np.delete(np.copy(a_x_train_set), obj=a_model_regression[i], axis=0)
        a_y_train_set_cpy = np.delete(np.copy(a_y_train_set), obj=a_model_regression[i], axis=0)
        
        a_x_test_cpy = a_x_train_set[i :  i + 1]
        a_y_test_cpy = a_y_train_set[i :  i + 1]
        
        model = LinearRegression().fit(a_x_train_set_cpy, a_y_train_set_cpy)
        
        a_y_pred = model.predict(a_x_test_cpy)
        
        SSE.append(mean_squared_error(a_y_test_cpy, a_y_pred))2

    plt.title = "Regression models comparison"
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")

    plt.plot(x, x, color="grey", linestyle="--", label="Y = x")
    #plt.ylim(-5, 5)
    #plt.xlim(-5, 5)
    plt.grid(True)
    
    plt.plot(np.array(y_training_set).reshape(-1), np.array(y_pred).reshape(-1), "o", color='green', label='Linear Regression')
    
    #for  i in range (len (y_pred)): 
    #    plt.text(y_train_set[i], y_pred[i], SSE[i], color='red', fontsize=10)
    

    plt.show()
