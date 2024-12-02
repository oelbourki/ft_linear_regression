import numpy as np
from numpy.linalg import inv
from math import sqrt

# class MyLinearRegression:
#     """
#     My personalized linear regression class to handle training, prediction, and evaluation.
#     """
#     history = []
#     def __init__(self, theta):
#         """
#         Initializes the linear regression model with theta parameters.
#         Args:
#             theta (list or np.ndarray): Initial weights, a vector of dimension (n_features + 1, 1).
#         """
#         if isinstance(theta, list):
#             theta = np.array(theta).reshape(-1, 1)
#         elif isinstance(theta, np.ndarray) and len(theta.shape) == 1:
#             theta = theta.reshape(-1, 1)
#         if not isinstance(theta, np.ndarray) or theta.shape[1] != 1:
#             raise ValueError("Theta must be a list or numpy array of shape (n, 1).")
#         self.theta = theta

#     def concat(self, X):
#         """
#         Adds a column of ones to the input features for the intercept term.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#         Returns:
#             np.ndarray: Features concatenated with a column of ones, shape (m, n+1).
#         """
#         if not isinstance(X, np.ndarray):
#             raise ValueError("X must be a numpy array.")
#         return np.c_[np.ones((X.shape[0], 1)), X]

#     def predict_(self, X):
#         """
#         Predicts the output using the model.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#         Returns:
#             np.ndarray: Predicted values of shape (m, 1).
#         """
#         X = self.concat(X)
#         return X.dot(self.theta)

#     def cost_elem_(self, X, Y):
#         """
#         Computes the cost element-wise.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             np.ndarray: Cost for each example.
#         """
#         m = Y.shape[0]
#         pred = self.predict_(X)
#         return (1 / (2 * m)) * np.power((pred - Y), 2)

#     def cost_(self, X, Y):
#         """
#         Computes the total cost for all examples.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             float: Total cost.
#         """
#         return np.sum(self.cost_elem_(X, Y))

#     # self.history = []  # To store theta at each epoch

#     def fit_(self, x, y, alpha=0.01, n_cycle=1000, save_interval=1000):
#         """
#         Fit the model using gradient descent to minimize the cost function.
#         Parameters:
#             x: np.array, input features
#             y: np.array, target values
#             alpha: float, learning rate
#             n_cycle: int, number of iterations (epochs)
#             save_interval: int, how often to save theta (in epochs)
#         """
#         m = len(x)
#         for epoch in range(n_cycle):
#             # Compute the prediction
#             y_pred = self.predict_(x)
            
#             # Compute the gradient of the cost function with respect to theta
#             gradient = (1/m) * np.dot(x.T, (y_pred - y))
            
#             # Update theta using gradient descent
#             self.theta -= alpha * gradient
            
#             # Save theta values at each save_interval epoch
#             if epoch % save_interval == 0:
#                 self.history.append(self.theta.copy())
#     def normalequation_(self, X, Y):
#         """
#         Solves for theta using the normal equation.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             np.ndarray: Updated theta values.
#         """
#         X = self.concat(X)
#         self.theta = inv(X.T.dot(X)).dot(X.T).dot(Y)
#         return self.theta

#     def mse_(self, X, Y):
#         """
#         Computes the Mean Squared Error (MSE).
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             float: MSE value.
#         """
#         error = self.predict_(X) - Y
#         return np.mean(np.power(error, 2))

#     def rmse_(self, X, Y):
#         """
#         Computes the Root Mean Squared Error (RMSE).
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             float: RMSE value.
#         """
#         return sqrt(self.mse_(X, Y))

#     def r2score_(self, X, Y):
#         """
#         Computes the R² score.
#         Args:
#             X (np.ndarray): Input features of shape (m, n).
#             Y (np.ndarray): True target values of shape (m, 1).
#         Returns:
#             float: R² score.
#         """
#         Y_pred = self.predict_(X)
#         ss_total = np.sum(np.power(Y - np.mean(Y), 2))
#         ss_residual = np.sum(np.power(Y - Y_pred, 2))
#         return 1 - (ss_residual / ss_total)
class MyLinearRegression(object):
    """    Description:        My personnal linear regression class to fit like a boss.   
    """
    def __init__(self, theta):
        """     Description:
        generator of the class, initialize self.
        Args:
            theta: has to be a list or a numpy array,
            it is a vector ofdimension (number of features + 1, 1).
            Raises:
           This method should noot raise any Exception.        
        """
        if (not isinstance(theta, list) and not isinstance(theta, np.ndarray)
        and theta.shape[1] != 1):
            print("error")
        if isinstance(theta, list):
            theta = np.array(theta).reshape(-1,1)
        self.theta = theta
    def setTheta(self, theta):
        if (not isinstance(theta, list) and not isinstance(theta, np.ndarray)
        and theta.shape[1] != 1):
            print("error")
        if isinstance(theta, list):
            theta = np.array(theta)
        self.theta = theta	
    def	concat(self, X):
        M = X.shape[0]
        ones = np.ones((M,1))
        X = np.concatenate((ones,X),axis=1)
        return X
    def predict_(self, X):
        X = self.concat(X)
        return X.dot(self.theta)
    def cost_elem_(self, X, Y):
        M = Y.shape[0]
        pred = X.dot(self.theta)
        return np.power((pred - Y),2,dtype=float) * (0.5 / M)
    def cost_(self, X, Y):
        return float(sum(self.cost_elem_(X, Y)))
    def grad_(self, error, X):
        X = self.concat(X)
        return (X.transpose()).dot(error)
    def mse_(self,X, Y):
        error = X.dot(self.theta) - Y
        M = float(Y.shape[0])
        return float(sum(np.power(error,2,dtype=float)) / M)
    def fit_(self, X, Y, alpha=1.6e-4,n_cycle=1000000):
        M = Y.shape[0]
        X = self.concat(X)
        for i in range(int(n_cycle)):
            error = X.dot(self.theta) - Y
            grad = (X.transpose()).dot(error)
            self.theta = self.theta - alpha * (1. / M ) * 0.5 * grad
            if i % 1000 == 0:
                print("cost: {}".format(self.mse_(X, Y)),end='\r')
        return self.theta
    def normalequation_(self, X, Y):
        X = self.concat(X)
        X_t = X.transpose()
        xx_t = X_t.dot(X).astype(np.int)
        X_ty = X_t.dot(Y)
        xx_ti = inv(xx_t)
        print(self.theta.shape)
        self.theta = (xx_ti.transpose()).dot(X_ty)
        print(self.theta.shape)
        return self.theta
    def rmse_(self, X, Y):
        return float(sqrt(self.mse_(X,Y)))
    def r2score_(self, X, Y):
        Yp = self.predict_(X)
        meanY = np.mean(Y)
        SStot = np.power(Y - meanY, 2,dtype=float)
        SSres = np.power(Y - Yp, 2,dtype=float)
        return float(1 - float(SSres / SStot))