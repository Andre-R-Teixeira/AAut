import itertools


import numpy as np 
from sklearn.linear_model import LinearRegression
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

    elements =  list(range(1, 14))
    #best_abs_diff = 1000
    count = 0
    T_r2 = 0
    error = 0
    Alpha= 10000
    
    
    #best_r = 0
    #best_train = []
    for i  in elements:
        permutations = calculate_permutations(15, i)

        for element in permutations:
            rows_to_cpy = element
            
            x_testing_set = x_train_set[rows_to_cpy, :]
            y_testing_set = y_train_set[rows_to_cpy, :]

            y_train_set_cpy = np.delete(y_train_set, rows_to_cpy, axis=0)
            x_train_set_cpy = np.delete(x_train_set, rows_to_cpy, axis=0)

            #poly_features = PolynomialFeatures(degree=6, include_bias=False)
            #x_train_poly = poly_features.fit_transform(x_train_set_cpy)
            #x_test_poly = poly_features.transform(x_testing_set)
        
            #model = LinearRegression() 

    
            #ridge_model = Ridge(alpha=Alpha)
            #ridge_model.fit(x_train_set_cpy, y_train_set_cpy)

            lasso_model = Lasso(alpha=Alpha)
            lasso_model.fit(x_train_set_cpy, y_train_set_cpy)
            
            #model.fit(x_train_set_cpy, y_train_set_cpy)
            #model.fit(x_train_poly, y_train_set_cpy)
            
            #r_sq = model.score(x_train_set_cpy, y_train_set_cpy)

            # Test prediction with the part of the training set
            #y_pred = model.predict(x_testing_set)
            #y_pred = model.predict(x_test_poly)
            #y_pred = ridge_model.predict(x_testing_set)
            y_pred = lasso_model.predict(x_testing_set)
            
            # Calculate Sum of square errors
            # print(f"\nRow removed to test {rows_to_cpy}")
            
            SSE = np.sum((y_testing_set - y_pred)**2)
            
            #print(f"SSE: {SSE}")
            
            # Calculate Sum of square total
            SST = np.sum((y_testing_set - np.mean(y_pred))**2)
            #print(f"SS_total: {SST}")
            
            # Calculate R2
            R2 = 1 - (SSE/SST)

            count = count + 1

            T_r2 = T_r2 + R2
            
            #print(f"R2 calculate {R2}")
            # print(f"R^2 : {R2}")
            
            #abs_diff = abs(1 - R2)
            #if (abs_diff < best_abs_diff):
                #best_abs_diff = abs_diff
                #best_train = rows_to_cpy
                #best_r = R2

        #print(f"Best train: {best_train} Best abs diff: {best_abs_diff} Best R2: {best_r}")
    
    error = T_r2/count
    print(f" Alpha: {Alpha}\n")
    print(f" Sum of Error: {error}")
# Itera

if __name__ == '__main__':
    main()