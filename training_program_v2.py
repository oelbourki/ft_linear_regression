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

    for i in range(n_iterations):
        predictions = estimate_price(mileage, theta0, theta1)
        error = predictions - price

        tmp_theta0 = learning_rate * np.sum(error) / m
        tmp_theta1 = learning_rate * np.sum(error * mileage) / m

        # Simultaneously update theta0 and theta1
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1

        # Optionally, print cost at intervals
        if i % 100 == 0:
            cost = compute_cost(mileage, price, theta0, theta1)
            print(f"Iteration {i}: Cost = {cost:.4f}")

    return theta0, theta1

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

    # Normalize the mileage (independent variable)
    mileage_mean = np.mean(mileage)
    mileage_std = np.std(mileage)
    mileage_normalized = (mileage - mileage_mean) / mileage_std

    # Hyperparameters
    learning_rate = 0.01  # Reduced learning rate
    n_iterations = 10000  # More iterations for better convergence

    # Train the model
    theta0, theta1 = gradient_descent(mileage_normalized, price, learning_rate, n_iterations)

    # Save the model parameters
    with open("model.json", "w") as file:
        json.dump({
            "theta0": theta0,
            "theta1": theta1,
            "mileage_mean": mileage_mean,
            "mileage_std": mileage_std
        }, file)

    print(f"Model trained successfully! Theta0: {theta0:.2f}, Theta1: {theta1:.2f}")

    # Plot the data and regression line (denormalized for plotting)
    mileage_range = np.linspace(min(mileage), max(mileage), 100)
    mileage_range_normalized = (mileage_range - mileage_mean) / mileage_std
    regression_line = theta0 + theta1 * mileage_range_normalized

    plt.scatter(mileage, price, color="blue", label="Data points")
    plt.plot(mileage_range, regression_line, color="red", label="Regression line")
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Linear Regression")
    plt.show()

if __name__ == "__main__":
    main()
