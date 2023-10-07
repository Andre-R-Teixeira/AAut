import sys
from sklearn.preprocessing import StandardScaler
import itertools
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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

""" The `PlotManager` class provides methods to initialize a plot, plot regression models, and display
 the plot with a legend. """

class PlotManager:
    def __init__(self):
        """
        The function initializes a plot with a title, x and y labels and a grid. It plots a line with a
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
        This function plots the real  y values against the predicted y values using a scatter plot with a specified color and model name as labels.

        :param y_real: The actual values of the dependent variable (y) in the regression model
        :param y_pred: The predicted values of the regression model
        :param model_name: The model_name parameter is a string that represents the name of the regression model being plotted.
        :param color: The color parameter is used to specify the color of the plotted points.
        """
        plt.plot(y_real, y_pred, "o", color=color, label=model_name)

        for i in range(len(y_real)):
            plt.plot([y_real[i], y_real[i]], [y_real[i], y_pred[i]], color=color, linestyle='--')
            plt.text(y_real[i], y_pred[i], str(i), fontsize=10, color='red')
        
    def show(self):
        """
        This function displays the plot and its legend.
        """
        plt.legend(loc="upper left")
        plt.show()

# The `Regression_Model_Tester` class is used to test a regression model by performing
# cross-validation and calculating the mean error of the model.
class Regression_Model_Tester:
    def __init__(self, X, Y, used_model, used_model_name, plot, color, parameters = None):
        """
        This function initializes the attributes of an object, including the
        input data, the used model, the model name, as well as empty lists for predicted values, real
        values, and prediction errors.

        :param X: This parameter represents the input data for the model.
        :param Y: This parameter represents the target variable or the dependent variable in a machine
        learning model. It is the variable that we are trying to predict or estimate based on the input
        variables X
        :param used_model: The `used_model` parameter is the machine learning model that will be used for
        prediction. 
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
        
        self.parameters = parameters
        

    @property
    def errors(self) -> np.array:
        """
        This function calculates the error of the model by taking the mean of the prediction error.
        :return: the mean of the `prediction_error` array.
        """
        if len(self.prediction_error) == 0:
            return sys.maxsize
        return np.mean(self.prediction_error)

    @errors.setter
    def errors_setter(self, errors): 
        self._previous_best_error = errors

    def _train_model(self, X_set, Y_set, i ):
        """
        This function trains a model on a training set, makes predictions on a test set and
        calculates the mean squared error between the predicted and actual values.
        
        :param X_set: X_set is a numpy array containing the input features for the training data. Each row
        of X_set represents a single training example, and each column represents a different feature
        :param Y_set: Y_set is the set of target values. It represents the true values that the model is trying to predict
        :param i: Thia parameter represents the index of the data point that is being used to split the data into training and testing sets 
        by excluding the data point at index "i" from the training set and using it as the test set
        :return: the mean squared error between the predicted y values and the actual y values.
        """
        
        
        x_train_set_cpy = np.delete(np.copy(X_set), i, axis=0)
        y_train_set_cpy = np.delete(np.copy(Y_set), i, axis=0)

        x_test = X_set[i : i + 1]
        y_test = Y_set[i : i + 1]

        self.used_model.fit(x_train_set_cpy, y_train_set_cpy)

        # Test prediction with the part of the training set
        y_pred = self.used_model.predict(x_test)

        y_pred, y_test

        return(mean_squared_error(y_test, y_pred), y_pred, y_test)


    def _cross_validation(self) -> None:                 
        """
        The function performs cross-validation by training a model on all columns of the input data and
        calculating the prediction error for each iteration.
        :return: None
        """
        for i in range(len(self.X)):
            SE, y_pred, y_real = self._train_model(np.copy(self.X), np.copy(self.Y), i)

            self.y_pred.append(y_pred)
            self.y_real.append(y_real)
            
            self.prediction_error.append(SE)

        return None

        
        

    def run_validation(self, number_of_column_to_remove = 0) -> np.ndarray:
        """
        This function performs cross-validation and optionally removes columns. Then plots the
        model and prints the mean error and the best combination of removed columns.
        
        :param number_of_column_to_remove: This parameter is an optional parameter that specifies the number of columns to remove during cross-validation. 
        :param plot_model: A boolean flag indicating whether or not to plot the model. If set to True, the
        model will be plotted, if set to False, the model will not be plotted
        :return: None
        """
        
        self._cross_validation()
        

        
        return self.errors
        
    def plot_model(self) -> None:
        """
        This function plots the predicted values against the real values.
        :return: None.
        """
        self.plot.plot_regression(
            np.array(self.y_real).reshape(-1),
            np.array(self.y_pred).reshape(-1),
            f"{self.used_model_name} : {self.errors}",
            color=self.color,
        )
        return None
    
    def model_logger(self): 
        print(f"Current Model: {self.used_model_name} current special parameters : {self.parameters}  current_error : {self.errors}")

def elastic_net(X_train, Y_train, plot, color, alphas=[0.09775999999999777], l1_ratios= [0.89], number_of_columns_to_remove = 0, plot_model=False) -> None:
    """
    This function performs Elastic Net regression on the given training data and returns the errors of the model.

    :param X_train: The training data for the independent variables 
    :param Y_train: The target variable for training the model
    :param plot: A boolean value indicating whether or not to plot the validation results
    :param color: This parameter is used to specify the color of the plot when visualizing the results.
    :param alpha: The alpha parameter controls the regularization strength of the Elastic Net model. It determines the balance between the L1 and L2 penalties.
    A higher alpha value leads to stronger regularization and can help prevent overfitting
    :param l1_ratio: This  parameter in ElasticNet is a value between 0 and 1 that determines the balance between L1 and L2 regularization
    :return: the errors from the Elastic Net model.
    """
    
    best_elastic_model = None
    
    for alpha in alphas: 
        for l1_ratio in l1_ratios:
            parameters = {"alpha": alpha, "l1_ratio": l1_ratio}
            
            elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)

            elastic_net_model = Regression_Model_Tester(
                X_train, 
                Y_train, 
                elastic_net, 
                "Elastic Net", 
                plot, 
                color, 
                parameters
            )

            if best_elastic_model is None:
                best_elastic_model = elastic_net_model

            current_sse = elastic_net_model.run_validation(number_of_column_to_remove=number_of_columns_to_remove)
            
            if (current_sse < best_elastic_model.errors): 
                best_elastic_model = elastic_net_model
    
    best_elastic_model.model_logger()
    
    if plot_model is True:
        best_elastic_model.plot_model()

    return None

def polynomial_model(X_train, Y_train, plot, color, degree=2, number_of_columns_to_remove = 0, plot_model=False) -> None:
    """
    This function fits a polynomial regression model of a specified degree to the
    given training data and evaluates its performance using a regression model tester.
    
    :param X_train: The training data for the independent variable
    :param Y_train: This parameter represents the dependent variable in the dataset. It is the variable that we are trying to predict
    :param plot: This parameter is a boolean value that determines whether or not to plot the
    regression line and data points. If set to True, the regression line and data points will be
    plotted, if set to False, no plot will be generated.
    :param color: This parameter is used to specify the color of the plot in the polynomial_model function. 
    :param degree: The degree parameter determines the degree of the polynomial features to be used in the polynomial regression model.
    :return: None.
    """

    polynomial_features = PolynomialFeatures(degree=degree, include_bias=True)
    
    x_poly = polynomial_features.fit_transform(X_train)

    polynomial_regression_model = Regression_Model_Tester(
        x_poly, 
        Y_train, 
        LinearRegression(), 
        "Polynomial Regression", 
        plot, 
        color
    )
    polynomial_regression_model.run_validation(number_of_column_to_remove=number_of_columns_to_remove)
    
    polynomial_regression_model.model_logger()
        
    if plot_model is True: 
        polynomial_regression_model.plot_model()

    return None

def ridge_model(X_train, Y_train, plot, color, alphas=[2.08710000000000001], number_of_columns_to_remove = 0, plot_model=False) -> None:
    """
    This function performs ridge regression on the given training data and prints the validation results.

    :param X_train: This parameter is the training data for the independent variables. 
    :param Y_train: This parameter represents the dependent variable in the training dataset. It is the variable that we are trying to predict.
    :return: None.
    """
    
    best_ridge_model = None
    
    for alpha in alphas:
    
        parameters = {"alpha": alpha}
    
        ridge_regression_model = Regression_Model_Tester(
            X_train,
            Y_train,
            Ridge(alpha=alpha, max_iter=10000),
            "Ridge Regression",
            plot,
            color, 
            parameters
        )
        
        if best_ridge_model is None:
            best_ridge_model = ridge_regression_model
        
        current_ridge_model_error = ridge_regression_model.run_validation(number_of_column_to_remove=number_of_columns_to_remove)

        if (current_ridge_model_error < best_ridge_model.errors): 
            best_ridge_model = ridge_regression_model
            
    if (plot_model is True): 
        best_ridge_model.plot_model()


    best_ridge_model.model_logger()

    return None

def lasso_model(X_train, Y_train, plot, color, alphas=[0.08710000000000001], number_of_columns_to_remove = 0, plot_model=False) -> None:
    """
    This function trains and tests a Lasso regression model using the provided training data.

    :param X_train: This parameter is the training data for the independent variables. 
    :param Y_train: This parameter represents the dependent variable in the dataset. It is the variable that we are trying to predict 
    :return: None.
    """
    
    best_lasso_model = None
    
    for alpha in alphas:
        
        parameters = {"alpha": alpha}
    
        lasso_regression_model = Regression_Model_Tester(
            X_train,
            Y_train,
            Lasso(alpha=alpha, max_iter=10000),
            "Lasso Regression",
            plot,
            color, 
            parameters
        )
        
        if best_lasso_model is None:
            best_lasso_model = lasso_regression_model
        
        current_lasso_model_error = lasso_regression_model.run_validation(number_of_column_to_remove=number_of_columns_to_remove)

        if (current_lasso_model_error < best_lasso_model.errors): 
            best_lasso_model = lasso_regression_model
            
    if (plot_model is True): 
        best_lasso_model.plot_model()
            
    best_lasso_model.model_logger()

    return None

def linear_model(X_train, Y_train, plot, color, number_of_columns_to_remove = 0, plot_model=False) -> None:
    """
    This function trains and tests a linear regression model using the given training data.

    :param X_train: This parameter is the training data for the independent variables in the linear regression model.
    :param Y_train: This parameter represents the dependent variable in the dataset. It is the variable that you are trying to predict.
    :return: None.
    """

    linear_regression_model = Regression_Model_Tester(
        X_train, 
        Y_train, 
        LinearRegression(), 
        "Linear Regression", 
        plot, 
        color
    )
    linear_regression_model.run_validation(number_of_column_to_remove=number_of_columns_to_remove)

    linear_regression_model.model_logger()

    if (plot_model is True):
        linear_regression_model.plot_model()

    return None

def cluster_kmeans(X, Y):
    data = np.column_stack((X, Y))
   
    kmeans = KMeans(n_clusters=2, random_state=42)
    cluster_labels = kmeans.fit_predict(data)


    return cluster_labels


def cluster_gaussian(X, Y):
    data = np.column_stack((X, Y))
   
    gaussian = GaussianMixture(n_components=2, random_state=0, max_iter=1000, tol=1e-12)
    cluster_labels = gaussian.fit_predict(data)

    return cluster_labels

def cluster_spectral(X, Y):
    #data = np.column_stack((X, Y))

   
    model = SpectralClustering(n_clusters=2, affinity="precomputed")
    cluster_labels = model.fit_predict(X, Y)

    return cluster_labels

def get_residuals(X, Y):
    model = LinearRegression()
    model.fit(X, Y)

    Y_PRED = model.predict(X)

    residuals = Y - Y_PRED
    mse = mean_squared_error(Y, Y_PRED)
    #print("MSE 1 model: ", mse)

    return Y_PRED


def main():
    """
    The main function loads training data and calls three different models: linear_model, ridge_model,
    and lasso_model.
    """

    plot = PlotManager()

    X = np.load("input_files/X_train_regression2.npy")
    Y = np.load("input_files/y_train_regression2.npy")

    data = np.column_stack((X, Y))

    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    Y_PRED = get_residuals(X, Y)

    # cluster_labels=cluster_gaussian(Y,Y_PRED)
    # cluster_labels=cluster_kmeans(X,Y)
    # data1,data2=cluster_spectral(X,Y)
    cluster_labels = cluster_gaussian(X, Y)

    data1 = data_scaled[cluster_labels == 0]
    data2 = data_scaled[cluster_labels == 1]

    
    DATA = [data1, data2]
    #plot = PlotManager(Y.tolist())


    #ridge_alpha = np.arange(0.1, 5, 0.1).tolist()
    #l_ratio = [0.89]
    #elastic_net_alpha = [0.09775999999999777] 
    ridge_alpha = [2.0]    
    #lasso_alpha = [0.08710000000000001]
    
    for i in range(2):
        data = DATA[i]
        x_train_set = data[:, :-1]
        y_train_set = data[:, -1]
        
        #linear_model(X_train=x_train_set,
            #             Y_train=y_train_set,
            #             plot= plot,
            #             color="blue", 
            #             number_of_columns_to_remove=0,
        #                 plot_model=True)

            #polynomial_model(X_train=x_train_set, 
            #                 Y_train=y_train_set,
            #                 plot=plot,
            #                 color="cyan",
            #                 degree=2)

        ridge_model(X_train=x_train_set,
                        Y_train=y_train_set,
                        plot=plot,
                        color="green",
                        alphas=ridge_alpha, 
                        plot_model=True)

            #lasso_model(X_train=x_train_set, 
            #            Y_train=y_train_set,
            #            plot=plot,
            #            color="red",
            #            alphas=lasso_alpha)
                
            #elastic_net(X_train=x_train_set,
            #            Y_train=y_train_set,
            #            plot=plot,
            #            color="purple", 
            #            alphas=elastic_net_alpha,
            #            l1_ratios=l_ratio)
    print("len1:", len(data1), "\nlen2:", len(data2))



   

    plot.show()


if __name__ == "__main__":
    main()
