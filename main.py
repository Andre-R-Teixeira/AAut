import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

## Load train data set
def load_train_data(x_train_set_filename, y_train_set_filename): 
    return np.load(x_train_set_filename), np.load(y_train_set_filename)



def main(): 
    files_folder = 'input_files/1_exercise/' 
    x_train_set, y_train_set = load_train_data(files_folder+'X_train_regression1.npy', files_folder+'y_train_regression1.npy')

    x_= PolynomialFeatures(degree=5,  include_bias=False).fit_transform(x_train_set)    
    
    print(f"Shape : {x_.shape} \nx:\n{x_}")
    
    model = LinearRegression() 
    model.fit(x_train_set, y_train_set)

    r_sq = model.score(x_train_set, y_train_set)
    
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")

    model = LinearRegression() 
    model.fit(x_, y_train_set)
    
    r_sq = model.score(x_train_set, y_train_set)
    
    print(f"intercept: {model.intercept_}")
    print(f"coefficients: {model.coef_}")


if __name__ == '__main__':
    main()