import sys

import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    LassoCV,
    RidgeCV,
    ElasticNet,
)

import time


# The `PlotManager` class provides methods to initialize a plot, plot regression models, and display
# the plot with a legend.
class PlotManager:
    def __init__(self):
        """
        The function initializes a plot with a title, x and y labels, and a grid, and plots a line with a
        specific range and style.
        """
        x = np.linspace(-10, 10, 1000)

        plt.title = "Regression models comparison"
        plt.xlabel("Predicted values")
        plt.ylabel("Real values")

        plt.plot(x, x, color="grey", linestyle="--", label="Y = x")
        plt.ylim(-5, 5)
        plt.xlim(-5, 5)
        plt.grid(True)

    def plot_regression(
        self, y_real=[], y_pred=[], model_name="No model name", color="blue"
    ):
        """
        The function `plot_regression` plots the real values (`y_real`) against the predicted values
        (`y_pred`) using a scatter plot with a specified color and model name as labels.

        :param y_real: The actual values of the dependent variable (y) in the regression model
        :param y_pred: The predicted values of the regression model
        :param model_name: The model_name parameter is a string that represents the name of the regression
        model being plotted. It is used as a label for the plot, defaults to No model name (optional)
        :param color: The color parameter is used to specify the color of the plotted points. It can be any
        valid color name or a hexadecimal color code, defaults to blue (optional)
        """
        plt.plot(y_real, y_pred, "o", color=color, label=model_name)

        for i in range(len(y_real)):
            #    plt.plot([y_real[i], y_real[i]], [y_real[i], y_pred[i]], color=color, linestyle='--')
            plt.text(y_real[i], y_pred[i], str(i), fontsize=10, color=color)

    def show(self):
        """
        The function displays a legend on a matplotlib plot and shows the plot.
        """
        plt.legend(loc="upper left")
        plt.show()


# The `Regression_Model_Tester` class is used to test a regression model by performing
# cross-validation and calculating the mean error of the model.
class Regression_Model_Tester:
    def __init__(self, X, Y, used_model, used_model_name, plot, color, remove_colums = False):
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
        self.plot = plot
        self.color = color
        
        self._previous_best_error = sys.maxsize
        self._previous_best_combination = []
        
        self.remove_colums =  remove_colums
        
    @property 
    def best_error_removing_colums(self): 
        return self._previous_best_error, self._previous_best_combination

    @property
    def errors(self) -> np.array:
        """
        The function calculates the error of the model by taking the mean of the prediction error.
        :return: the mean of the `prediction_error` array.
        """

        return np.mean(self.prediction_error)

    @errors.setter
    def errors_setter(self, errors): 
        self._previous_best_error = errors

    def _train_model(self, X_set, Y_set, i):

        x_train_set_cpy = np.delete(np.copy(X_set), i, axis=0)
        y_train_set_cpy = np.delete(np.copy(Y_set), i, axis=0)

        x_test = X_set[i : i + 1]
        y_test = Y_set[i : i + 1]

        self.used_model.fit(x_train_set_cpy, y_train_set_cpy)

        # Test prediction with the part of the training set
        y_pred = self.used_model.predict(x_test)

        self.y_pred.append(y_pred)
        self.y_real.append(y_test)

        return( mean_squared_error(y_test, y_pred))
        
        
    def _cross_validation(self) -> None:
        """
        The `cross_validation` function performs cross-validation by splitting the data into training and
        testing sets, fitting the model on the training set, and evaluating the model's performance on the
        testing set.
        :return: None.
        """

        max_4_combinations = []
        
        
        
        #print(f"len : len(self.X) {(np.shape(self.X)[1])}")
        if self.remove_colums is not False:
            for lenght  in range (1, 6): 
                for combo in itertools.combinations(range(10), lenght - 1):
                    max_4_combinations.append(combo)
                
                for j in max_4_combinations: 
                    total_error = []

                    x_train_set_removed_colums = np.delete(np.copy(self.X), j, axis=1)
                    
                    for i in range (len(self.X)):
                        SE = self._train_model(x_train_set_removed_colums, np.copy(self.Y), i)
        
                        total_error.append(SE)
                        
                    #print(f"total erros  {np.mean(total_error)}")
                        
                    if (np.mean(total_error) < self._previous_best_error):
                        self._previous_best_error = np.mean(total_error)
                        self._previous_best_combination = j 
                        self.prediction_error = total_error                        


                        

        else:
            for i in range(len(self.X)):
                SE  = self._train_model(np.copy(self.X), np.copy(self.Y), i)

                self.prediction_error.append(SE)
            
        return None

    def run_validation(self) -> None:
        """
        The function "run_validation" performs cross-validation and prints the mean error.
        :return: None.
        """
        self._cross_validation()
        # self.plot_model()
        print(f"Calculating mean of error for {self.used_model_name} : {self.errors} colums remove {self._previous_best_combination}")
        return None

    def plot_model(self) -> None:
        """
        The function `plot_model` plots the predicted values against the real values.
        :return: None.
        """
        self.plot.plot_regression(
            np.array(self.y_real).reshape(-1),
            np.array(self.y_pred).reshape(-1),
            f"{self.used_model_name} : {self.errors}",
            color=self.color,
        )
        return None


def elastic_net(X_train, Y_train, plot, color, alpha, l1_ratio) -> np.array:
    """
    The function ElasticNet performs Elastic Net regression on the given training data and returns the
    errors of the model.

    :param X_train: The training data for the independent variables (features)
    :param Y_train: The target variable for training the model
    :param plot: A boolean value indicating whether or not to plot the validation results
    :param color: The "color" parameter is used to specify the color of the plot when visualizing the
    results. It can be any valid color value, such as "red", "blue", "#FF0000" (hexadecimal color code),
    or "rgb(255, 0, 0)" (
    :param alpha: The alpha parameter controls the regularization strength of the Elastic Net model. It
    determines the balance between the L1 and L2 penalties. A higher alpha value leads to stronger
    regularization and can help prevent overfitting
    :param l1_ratio: The l1_ratio parameter in ElasticNet is a value between 0 and 1 that determines the
    balance between L1 and L2 regularization
    :return: the errors from the Elastic Net model.
    """
    elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

    elastic_net_model = Regression_Model_Tester(
        X_train, Y_train, elastic_net, "Elastic Net", plot, color, True
    )

    elastic_net_model.run_validation()

    return elastic_net_model.errors


def polynomial_model(X_train, Y_train, plot, color) -> None:
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

    polynomial_features = PolynomialFeatures(degree=3, include_bias=True)
    x_poly = polynomial_features.fit_transform(X_train)

    polynomial_regression_model = Regression_Model_Tester(
        x_poly, Y_train, LinearRegression(), "Polynomial Regression", plot, color
    )
    polynomial_regression_model.run_validation()

    return None


def ridge_model(X_train, Y_train, plot, color, alpha) -> np.array:
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
    ridge_regression_model = Regression_Model_Tester(
        X_train,
        Y_train,
        Ridge(alpha=alpha, max_iter=10000),
        "Ridge Regression",
        plot,
        color,
        True
    )
    ridge_regression_model.run_validation()

    return ridge_regression_model.errors


def lasso_model(X_train, Y_train, plot, color, alpha) -> np.array:
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
    lasso_regression_model = Regression_Model_Tester(
        X_train,
        Y_train,
        Lasso(alpha=alpha, max_iter=10000),
        "Lasso Regression",
        plot,
        color,
        True , 
    )
    lasso_regression_model.run_validation()

    return lasso_regression_model.errors


def linear_model(X_train, Y_train, plot, color) -> None:
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

    linear_regression_model = Regression_Model_Tester(
        X_train, Y_train, LinearRegression(), "Linear Regression", plot, color, True
    )
    linear_regression_model.run_validation()

    return None


def main():
    """
    The main function loads training data and calls three different models: linear_model, ridge_model,
    and lasso_model.
    """
    smallest_error = sys.maxsize
    best_ratio = 0
    best_alpha = 0

    ridge_smallest_error = sys.maxsize
    ridge_best_alpha = 0

    lasso_models_smallest_error = sys.maxsize
    lasso_models_best_alpha = 0

    plot = PlotManager()

    l_ratio = 0.89
    elastic_net_alpha = 0.09775999999999777 
    ridge_alpha = 2.08710000000000001    
    lasso_alpha = 0.08710000000000001

    f = open("errors.txt", "w")
    f.write("Errors for regression models:\n")

    x_train_set = np.load("input_files/1_exercise/X_train_regression1.npy")
    y_train_set = np.load("input_files/1_exercise/y_train_regression1.npy")

    linear_model(x_train_set, y_train_set, plot, "blue")

    polynomial_model(x_train_set, y_train_set, plot, "cyan")

    ridge_model(x_train_set, y_train_set, plot, "green", ridge_alpha)

    lasso_model(x_train_set, y_train_set, plot, "red", lasso_alpha)
        
    elastic_net(x_train_set, y_train_set, plot, "purple", elastic_net_alpha, l_ratio)

    # plot.show()
    f.close()


if __name__ == "__main__":
    main()

