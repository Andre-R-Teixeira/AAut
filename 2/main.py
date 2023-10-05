import sys 

import random 

import itertools

import numpy as  np 

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression 

#from try_later import try_later


class PlotManager:
    def __init__(self):
        """
        The function initializes a plot with a title, x and y labels and a grid. It plots a line with a
        specific range and style.
        """


    def plot_regression(
        self, y_real=[], y_pred=[], model_name="No model name", color="blue"):
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


class PointClassifier: 
    def __init__ (self, model, model_name, x_train_set, y_train_set, min_points = 15, max_iterations = 30): 

        # sklearn model used to classify 
        self.model = model
        self.model_name = model_name

        ##
        self.x_train_set = x_train_set
        self.y_train_set = y_train_set

        # max number of points per model
        self.min_points = min_points
        self.max_iterations  = max_iterations 

        self.a_sse = []
        self.a_mse = []
    
        self.b_sse = []
        self.b_mse =  []
        
        self.sum_sse = []
        self.sum_mse = []

        self.points = []

    def run_validation(self): 
        self._random_combinations()
        
        self.MSE
        
        self.SSE

    @property
    def MSE(self):
        #print(f"Best a value : {min(self.a_mse)possible_comb points are : {self.points [ self.a_mse.index( min(self.a_mse) ) ] [0] }")
        #print(f"Best b value : {min(self.b_mse)} points are : {self.points [ self.b_mse.index( min(self.b_mse) ) ] [1] }")
        print(f"Best total value : {min(self.sum_mse)} a points are : {self.points[self.sum_mse.index(min(self.sum_mse))] [0]} b points are : {self.points[self.sum_mse.index(min(self.sum_mse))] [1] }")
        
    @property
    def SSE(self): 
       
        print(f"Best total value : {min(self.sum_sse)} a points are : {self.points[self.sum_sse.index(min(self.sum_sse))] [0]} b points are : {self.points[self.sum_sse.index(min(self.sum_sse))] [1] }")
        
    def _random_combinations(self):
        for number_of_points in range (self.min_points, len(self.x_train_set) - self.min_points):
            #possible_a_points = list(itertools.com)
            
            for  iteration in range (self.max_iterations): 
                ## random points for group a
                a_model_points  = random.sample(range(1, len(self.x_train_set)), number_of_points)
                # B points are all those not in A
                b_model_points =  [x for x in [i for i in range(100)] if x not in a_model_points]

                self.points.append([a_model_points, b_model_points])

                a_x_set = np.delete(np.copy(self.x_train_set), obj=b_model_points, axis = 0)
                b_x_set = np.delete(np.copy(self.x_train_set), obj=a_model_points, axis = 0)

                a_y_set = np.delete(np.copy(self.y_train_set), obj=b_model_points, axis = 0)
                b_y_set = np.delete(np.copy(self.y_train_set), obj=a_model_points, axis = 0)

                a_model =  self.model_tester(self.model, a_x_set, a_y_set)
                b_model =  self.model_tester(self.model, b_x_set, b_y_set)

                a_model.cross_validation()
                b_model.cross_validation()

                self.a_sse.append(a_model.SSE)
                self.a_mse.append(a_model.MSE)

                self.b_sse.append(b_model.SSE)
                self.b_mse.append(b_model.MSE)

                self.sum_sse.append(a_model.SSE + b_model.SSE)
                self.sum_mse.append(a_model.MSE + b_model.MSE) 


            
    class model_tester: 
        def __init__ (self, model, x_set, y_set): 
            """
            The above function is a constructor that initializes the model, x_set, and y_set variables, and also
            initializes an empty list called SSE.
            
            :param model: The model parameter refers to the machine learning model that will be used for
            training and prediction. It could be any model such as linear regression, decision tree, or neural
            network
            :param x_set: The x_set parameter is a set of input data that will be used to train the model. It
            typically consists of a matrix or array where each row represents a single input sample and each
            column represents a feature or attribute of that sample
            :param y_set: The y_set parameter is a set of target values or labels that correspond to the x_set
            parameter. It is used for training and evaluating the model
            """
            self.model = model
            self.x_set = x_set
            self.y_set = y_set

            self.sse = []


        @property
        def SSE(self): 
            """
            The above function is a property decorator that returns the sum of the SSE attribute.
            :return: The property "SSE" is being returned.
            """
            
            return np.sum(self.sse)

            

        @property 
        def MSE(self):
            """
            The above function calculates the mean squared error (MSE) by taking the mean of the sum of squared
            errors (SSE).
            :return: The property `MSE` is being returned, which is the mean of the sum of squared errors
            (`SSE`).
            """
            
            return np.mean(self.sse)          

        def  cross_validation(self): 
            """
            The function performs cross-validation by training a model on a subset of the data and testing it on
            a single data point, and then calculates the sum of squared errors (SSE) between the predicted and
            actual values.
            """
            for i in range (len (self.x_set)): 
                x_train_set = np.delete(np.copy(self.x_set), obj=i, axis=0)
                y_train_set = np.delete(np.copy(self.y_set), obj=i, axis=0)

                x_test_set = self.x_set[i :  i + 1]
                y_real = self.y_set[i :  i + 1]

                model = self.model.fit(x_train_set, y_train_set)

                y_pred = model.predict(x_test_set)
                
                self.sse.append((y_pred - y_real)**2)



def main():
    x_training_set = np.load('input_files/X_train_regression2.npy')
    y_training_set = np.load('input_files/y_train_regression2.npy')
    x_test_set = np.load('input_files/X_test_regression2.npy')
    
    point_classifier = PointClassifier(
       model=LinearRegression(),
       model_name="Linear Regression",
       x_train_set= np.copy(x_training_set),
       y_train_set= np.copy(y_training_set),
       min_points=20,
       max_iterations=200
   )
    
    point_classifier.run_validation()
    
if __name__ == '__main__': 
    main()