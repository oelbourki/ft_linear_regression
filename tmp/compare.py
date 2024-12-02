import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from my_linear_regression import MyLinearRegression  # Assuming your class is in a separate file

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Initialize theta with small random values
theta_initial = np.random.randn(2, 1) * np.sqrt(2. / diabetes_X_train.shape[0])

# Instantiate your custom model
my_model = MyLinearRegression(theta_initial)

# Train the model using the training sets
my_model.fit_(diabetes_X_train, diabetes_y_train, alpha=1e-5, n_cycle=10000)

# Make predictions using the testing set
my_y_pred = my_model.predict_(diabetes_X_test)

# The coefficients from your model
print("MyLinearRegression Coefficients: \n", my_model.theta)

# The mean squared error for your model
my_mse = mean_squared_error(diabetes_y_test, my_y_pred)
print(f"MyLinearRegression Mean Squared Error: {my_mse:.2f}")

# The coefficient of determination (R²) for your model
my_r2 = r2_score(diabetes_y_test, my_y_pred)
print(f"MyLinearRegression R² Score: {my_r2:.2f}")

# --------- Now using sklearn's LinearRegression ---------
from sklearn.linear_model import LinearRegression

# Create and train the sklearn model
sklearn_model = LinearRegression()
sklearn_model.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the sklearn model
sklearn_y_pred = sklearn_model.predict(diabetes_X_test)

# The coefficients from sklearn's model
print("\nsklearn LinearRegression Coefficients: \n", sklearn_model.coef_)

# The mean squared error for sklearn's model
sklearn_mse = mean_squared_error(diabetes_y_test, sklearn_y_pred)
print(f"sklearn LinearRegression Mean Squared Error: {sklearn_mse:.2f}")

# The coefficient of determination (R²) for sklearn's model
sklearn_r2 = r2_score(diabetes_y_test, sklearn_y_pred)
print(f"sklearn LinearRegression R² Score: {sklearn_r2:.2f}")

# --------- Plot outputs ---------
# Plotting for MyLinearRegression
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, my_y_pred, color="blue", linewidth=3, label="MyLinearRegression")

# Plotting for sklearn LinearRegression
plt.plot(diabetes_X_test, sklearn_y_pred, color="red", linewidth=3, label="sklearn LinearRegression")

plt.xlabel('Feature: BMI')
plt.ylabel('Target: Diabetes Progression')
plt.legend()
plt.title('Comparison of Linear Regression Models')
plt.show()
