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
    files_folder = 'input_files/1_exercise/' 
    x_train_set, y_train_set = load_train_data(files_folder+'X_train_regression1.npy', files_folder+'y_train_regression1.npy')

    T_r2 = 0
    error = 0
    
    
    #poly = PolynomialFeatures(degree=1)
   
    #x_train_set = poly.fit_transform(x_train_set)

    for i in range(15):
            
        x_copy= np.copy(x_train_set)
        y_copy = np.copy(y_train_set)

        x_train_set_cpy = np.delete(x_copy, i, axis=0)
        y_train_set_cpy = np.delete(y_copy, i, axis=0)

        x_test  = x_train_set[i : i+1]
        y_test = y_train_set[i : i+1] 

        #model = Lasso(alpha= )
        #model = Ridge(alpha= )
        model = LinearRegression() 
        model.fit(x_train_set_cpy, y_train_set_cpy)

        
        # Test prediction with the part of the training set
        y_pred = model.predict(x_test)

        SE = mean_squared_error(y_test, y_pred)
            

        T_r2 = T_r2 + SE
            
    
    error = T_r2/15
    #print(f" Alpha: {Alpha}\n")
    print(f" Mean of Error: {error}")

if __name__ == '__main__':
    main()