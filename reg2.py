import itertools
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


## Load train data set
def load_train_data(x_train_set_filename, y_train_set_filename): 
    return np.load(x_train_set_filename), np.load(y_train_set_filename)

def calculate_permutations(list_size, size_of_combination):
    # Define size of the list for permutations   
    elements = list(range(list_size))

    # Generate all the permutations of 3 elements from the list and store them in a NumPy array
    return np.array(list(itertools.combinations(elements, size_of_combination)))

def main(): 
    X_train = np.load('X_train_regression2.npy')
    y_train= np.load('y_train_regression2.npy')

    beta= 0.60

    #split_point = len(X_train) // 2
    split_point = int(len(X_train) * beta) 


    X_model1_train = X_train[:split_point]
    y_model1_train = y_train[:split_point]

    X_model2_train = X_train[split_point:]
    y_model2_train = y_train[split_point:]

    T_r2_1 = 0
    T_r2_2 = 0
    error_1 = 0
    error_2 = 0
    Best_error_1 = 1000
    Best_error_2 = 1000

    for i in range(15):
            
        x_copy1= np.copy(X_model1_train)
        y_copy1 = np.copy(y_model1_train)

        x_copy2= np.copy(X_model2_train)
        y_copy2 = np.copy(y_model2_train)

        x_train1_set_cpy = np.delete(x_copy1, i, axis=0)
        y_train1_set_cpy = np.delete(y_copy1, i, axis=0)

        x_train2_set_cpy = np.delete(x_copy2, i, axis=0)
        y_train2_set_cpy = np.delete(y_copy2, i, axis=0)


        x_test1  = x_train1_set_cpy[i : i+1]
        y_test1 = y_train1_set_cpy[i : i+1] 

        x_test2  = x_train2_set_cpy[i : i+1]
        y_test2 = y_train2_set_cpy[i : i+1] 


        model1 = LinearRegression() 
        model1.fit(x_train1_set_cpy, y_train1_set_cpy)

        model2 = LinearRegression() 
        model2.fit(x_train2_set_cpy, y_train2_set_cpy)

            
        # Test prediction with the part of the training set
        y_pred1 = model1.predict(x_test1)
        SSE1= mean_squared_error(y_test1, y_pred1)
        T_r2_1 = T_r2_1 + SSE1

        y_pred2 = model2.predict(x_test2)
        SSE2= mean_squared_error(y_test2, y_pred2)
        T_r2_2 = T_r2_2 + SSE2

            
            
    error_1 = T_r2_1/15
    if error_1 < Best_error_1:
        Best_error_1 = error_1

    error_2 = T_r2_2/15
    if error_2 < Best_error_2:
        Best_error_2 = error_2
      

    print(f" Beta: {beta}")
    print(f" Mean of Error 1: {Best_error_1}")
    print(f" Mean of Error 2: {Best_error_2}")

if __name__ == '__main__':
    main()

