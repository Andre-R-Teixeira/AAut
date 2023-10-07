from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools


 
## Load train data set
def load_train_data(x_train_set_filename, y_train_set_filename): 
    return np.load(x_train_set_filename), np.load(y_train_set_filename)

def calculate_permutations(list_size, size_of_combination):
    # Define size of the list for permutations   
    elements = list(range(list_size))

    # Generate all the permutations of 3 elements from the list and store them in a NumPy array
    return np.array(list(itertools.combinations(elements, size_of_combination)))

def models(x,y,lenght): 
    
    x_train_set= x
    y_train_set = y 


    T_r2 = 0
    error = 0
    Best_error = 1000

    T_r2 = 0 
    for i in range(lenght-1):
            
        x_copy= np.copy(x_train_set)
        y_copy = np.copy(y_train_set)

        x_train_set_cpy = np.delete(x_copy, i, axis=0)
        y_train_set_cpy = np.delete(y_copy, i, axis=0)


        x_test  = x_train_set_cpy[i : i+1]
        y_test = y_train_set_cpy[i : i+1] 

        #model = Lasso(alpha=Alpha)
        #model = Ridge(alpha=alphas)
        model = LinearRegression() 
        model.fit(x_train_set_cpy, y_train_set_cpy)

        
        # Test prediction with the part of the training set
        y_pred = model.predict(x_test)

        SE = mean_squared_error(y_test, y_pred)
            

        T_r2 = T_r2 + SE

            
            
    error = T_r2/lenght
    if error < Best_error:
        Best_error = error
        return(Best_error)
        
    
    
                




def main():
    
    x = np.load("input_files/X_train_regression2.npy")
    y = np.load("input_files/y_train_regression2.npy")

    # Combine X and y for each model
    data = np.column_stack((x, y))
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=2, n_init=100, random_state=None)
    kmeans.fit(data)

    # Separate data points based on the cluster labels
    model1_data = data[kmeans.labels_ == 0]
    model2_data = data[kmeans.labels_ == 1]

    # Extract X and y for each model
    X1_train_set = model1_data[:, :-1]
    Y1_train_set  = model1_data[:, -1]

    X2_train_set  = model2_data[:, :-1]
    Y2_train_set  = model2_data[:, -1]

    num_members_cluster1 = np.bincount(kmeans.labels_)[0]
    num_members_cluster2 = np.bincount(kmeans.labels_)[1]

    print(f"\nNumber of elements 1: {num_members_cluster1}\n")
    print(f"\nNumber of elements 2: {num_members_cluster2}\n")

    #for Alpha in np.arange(0.01, 10, 0.01):
    best_error_model1 = models(X1_train_set,Y1_train_set ,num_members_cluster1)
        
    best_error_model2 = models(X2_train_set,Y2_train_set, num_members_cluster2)
        
        
    

        #print(f" Alpha: {Alpha}\n")
    print(f" Mean of Error 1: {best_error_model1}")
    print(f" Mean of Error 2: {best_error_model2}")
            
    
        
if __name__ == "__main__":
    main()
