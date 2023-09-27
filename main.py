import itertools


import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV

class PlotManager:
    def __init__(self):
        pass
    
    
# The `Regression_Model_Tester` class is used to test a regression model by performing
# cross-validation and calculating the mean error of the model.
class Regression_Model_Tester:
    def __init__ (self, X, Y, used_model, used_model_name ):
        """
        The above function is a constructor that initializes the attributes of an object, including the
        input data, the used model, and the model name, as well as empty lists for predicted values, real
        values, and prediction errors.
        
        :param X: The X parameter represents the input data for the model. It could be a matrix or an array
        containing the features or independent variables used for prediction
        :param Y: The Y parameter represents the target variable or the dependent variable in a machine
        learning model. It is the variable that we are trying to predict or estimate based on the input
        variables X
        :param used_model: The `used_model` parameter is the machine learning model that will be used for
        prediction. It could be any model such as linear regression, decision tree, random forest, etc. The
        specific model will be passed as an argument when creating an instance of this class
        :param used_model_name: The name of the machine learning model that is being used
        """
        self.X = X
        self.Y = Y
        self.used_model = used_model
        self.used_model_name = used_model_name
        
        self.y_pred = []
        self.y_real = []
        self.prediction_error = []
        

    @property
    def errors(self) ->  np.array:
        """
        The function calculates the error of the model by taking the mean of the prediction error.
        :return: the mean of the `prediction_error` array.
        """
        
        return np.mean(self.prediction_error)
        
    def __cross_validation (self) -> None: 
        """
        The `cross_validation` function performs cross-validation by splitting the data into training and
        testing sets, fitting the model on the training set, and evaluating the model's performance on the
        testing set.
        :return: None.
        """
        
        for i in range (len(self.X)):            

            x_train_set_cpy = np.delete(np.copy(self.X), i, axis=0)
            y_train_set_cpy = np.delete(np.copy(self.Y), i, axis=0)
            
            x_test  = self.X[i : i+1]
            y_test = self.Y[i : i+1] 
                
            self.used_model.fit(x_train_set_cpy, y_train_set_cpy)
                    
            # Test prediction with the part of the training set
            y_pred = self.used_model.predict(x_test)
            
            self.y_pred.append(y_pred)
            self.y_real.append(y_test)

            SE = mean_squared_error(y_test, y_pred)
            self.prediction_error.append(SE)

        return None
    
    def run_validation (self) -> None: 
        """
        The function "run_validation" performs cross-validation and prints the mean error.
        :return: None.
        """
        self.__cross_validation()
        print(f"Calculating mean of error for {self.used_model_name} : {self.errors}")
        return None
        

def polynomial_model(X_train, Y_train) -> None:
    """
    The `polynomial_model` function creates a polynomial regression model by transforming the input data
    into a polynomial form and then fitting the model on the transformed data.
    :param X_train: The X_train parameter represents the input data for the model. It could be a matrix or
    an array containing the features or independent variables used for prediction
    :param Y_train: The Y_train parameter represents the target variable or the dependent variable in a
    machine learning model. It is the variable that we are trying to predict or estimate based on the input
    variables X
    :return: None.
    """
    
    polynomial_features = PolynomialFeatures(degree = 2)
    x_poly = polynomial_features.fit_transform(X_train)
    
    polynomial_regression_model = Regression_Model_Tester (
        x_poly, 
        Y_train, 
        LinearRegression(), 
        "Polynomial Regression"
    )
    polynomial_regression_model.run_validation()
    
    return None

def linear_model(X_train, Y_train) -> None: 
    """
    The function `linear_model` trains and tests a linear regression model using the given training
    data.
    
    :param X_train: The X_train parameter is the training data for the independent variables in your
    linear regression model. It should be a 2-dimensional array or dataframe where each row represents a
    sample and each column represents a feature
    :param Y_train: The parameter Y_train represents the target variable or the dependent variable in
    your dataset. It is the variable that you are trying to predict or model using the independent
    variables (X_train)
    :return: None.
    """
    
    linear_regression_model = Regression_Model_Tester (
        X_train, 
        Y_train, 
        LinearRegression(), 
        "Linear Regression"
    )
    linear_regression_model.run_validation()
    
    return None
    
def ridge_model(X_train, Y_train) -> None:
    """
    The function `ridge_model` performs ridge regression on the given training data and prints the
    validation results.
    
    :param X_train: The X_train parameter is the training data for the independent variables. It should
    be a matrix or dataframe with shape (n_samples, n_features), where n_samples is the number of
    samples or observations and n_features is the number of independent variables or features
    :param Y_train: The parameter Y_train represents the target variable or the dependent variable in
    your training dataset. It is the variable that you are trying to predict or model using the
    independent variables (X_train)
    :return: None.
    """
    ridge_regression_model = Regression_Model_Tester (
        X_train, 
        Y_train, 
        Ridge(alpha = .9), 
        "Ridge Regression"
    )
    ridge_regression_model.run_validation()
    
    return None
    
def lasso_model(X_train, Y_train) -> None:
    """
    The function `lasso_model` trains and tests a Lasso regression model using the provided training
    data.
    
    :param X_train: The parameter X_train is the training data for the independent variables. It should
    be a matrix or dataframe with shape (n_samples, n_features), where n_samples is the number of
    samples or observations and n_features is the number of independent variables or features
    :param Y_train: The parameter Y_train represents the target variable or the dependent variable in
    your dataset. It is the variable that you are trying to predict or model using the independent
    variables (X_train)
    :return: None.
    """
    lasso_regression_model = Regression_Model_Tester (
        X_train, 
        Y_train, 
        Lasso(), 
        "Lasso Regression"
    )
    lasso_regression_model.run_validation()
    
    return None


def main(): 
    """
    The main function loads training data and calls three different models: linear_model, ridge_model,
    and lasso_model.
    """
    
    x_train_set = np.load ('input_files/1_exercise/X_train_regression1.npy')
    y_train_set = np.load ('input_files/1_exercise/y_train_regression1.npy')
    
    linear_model(x_train_set, y_train_set)

    ridge_model(x_train_set, y_train_set)
    
    lasso_model(x_train_set, y_train_set)

if __name__ == '__main__':
    main()
    