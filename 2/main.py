from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import numpy as np
import itertools

def models(x,y,lenght,alphas): 
    
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

        model = Lasso(alpha=alphas)
        #model = Ridge(alpha=alphas)
        #model = LinearRegression() 
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

    model = LinearRegression() 
    model.fit(x, y)

        
    # Test prediction with the part of the training set
    y_pred = model.predict(x)

    residuals = y - y_pred


    kmeans = KMeans(n_clusters=2, n_init=100, random_state=None)
    kmeans.fit(residuals.reshape(-1, 1))

    cluster1_indices = np.where(kmeans.labels_ == 0)
    cluster2_indices = np.where(kmeans.labels_ == 1)

    # Separate data points based on the cluster labels
    cluster1_data = residuals[cluster1_indices]
    cluster2_data = residuals[cluster2_indices]


    num_members_cluster1 = np.bincount(kmeans.labels_)[0]
    num_members_cluster2 = np.bincount(kmeans.labels_)[1]

    print(f"\nNumber of elements 1: {num_members_cluster1}\n")
    print(f"\nNumber of elements 2: {num_members_cluster2}\n")

    
    Alpha=0.0001
    best_error_model1 = models(x[cluster1_indices],y[cluster1_indices] ,num_members_cluster1,Alpha)
        
    best_error_model2 = models(x[cluster2_indices],y[cluster2_indices], num_members_cluster2,Alpha)
        
        
    

    print(f" Alpha: {Alpha}\n")
    print(f" Mean of Error 1: {best_error_model1}")
    print(f" Mean of Error 2: {best_error_model2}")
            
    
        
if __name__ == "__main__":
    main()
