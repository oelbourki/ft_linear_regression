import numpy as np
import json
import matplotlib.pyplot as plt

def estimate_price(mileage, theta0, theta1):
    """Calculate the estimated price based on the given mileage."""
    return theta0 + theta1 * mileage

def compute_cost(mileage, price, theta0, theta1):
    """Compute Mean Squared Error for given parameters."""
    m = len(price)
    predictions = theta0 + theta1 * mileage
    return np.sum((predictions - price) ** 2) / (2 * m)

def gradient_descent(mileage, price, learning_rate, n_iterations):
    """Perform gradient descent to find the optimal theta values."""
    m = len(price)
    theta0, theta1 = 0, 0

    for _ in range(n_iterations):
        predictions = estimate_price(mileage, theta0, theta1)
        error = predictions - price

        tmp_theta0 = learning_rate * np.sum(error) / m
        tmp_theta1 = learning_rate * np.sum(error * mileage) / m

        # Simultaneously update theta0 and theta1
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

    return theta0, theta1

# def gradient_descent(mileage, price, learning_rate, n_iterations):
#     n_samples = len(price)
#     n_features = 1
#     y = price
#     weights = np.zeros(n_features)
#     weights = weights.reshape(-1,1)
#     bias = 0
#     X = mileage.reshape(-1,1)
#     for _ in range(n_iterations):
#         y_pred = X.dot(weights) + bias
#         dw = -(2 / n_samples) * X.T.dot(y - y_pred)
#         db = -(2 / n_samples) * np.sum(y - y_pred)

#         weights -= learning_rate * dw
#         bias -= learning_rate * db
#     # print
#     return weights[0], weights[1]
def plot_regression(mileage, price, theta0, theta1):
    """Plot the data points and the regression line."""
    plt.scatter(mileage, price, color="blue", label="Data points")
    regression_line = theta0 + theta1 * mileage
    plt.plot(mileage, regression_line, color="red", label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Linear Regression")
    plt.show()

def main():
    # Load dataset
    try:
        data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
        mileage = data[:, 0]
        price = data[:, 1]
    except FileNotFoundError:
        print("Dataset file not found. Please provide 'data.csv'.")
        return

    # Hyperparameters
    learning_rate = 0.0001
    n_iterations = 1000

    # Train the model
    theta0, theta1 = gradient_descent(mileage, price, learning_rate, n_iterations)

    # Save the model parameters
    with open("model.json", "w") as file:
        json.dump({"theta0": theta0, "theta1": theta1}, file)

    print(f"Model trained successfully! Theta0: {theta0:.2f}, Theta1: {theta1:.2f}")

    # Plot the data and regression line
    plot_regression(mileage, price, theta0, theta1)

    # Calculate and display precision (cost)
    cost = compute_cost(mileage, price, theta0, theta1)
    print(f"Final Cost (MSE): {cost:.2f}")

if __name__ == "__main__":
    main()
