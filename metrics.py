import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def estimate_price(mileage, theta0, theta1, mileage_mean, mileage_std):
    """Calculate the estimated price based on the given mileage."""
    # Apply the same scaling used during training
    mileage_normalized = (mileage - mileage_mean) / mileage_std
    return theta0 + theta1 * mileage_normalized

def main():
    # Load dataset
    try:
        data = np.loadtxt("data.csv", delimiter=",", skiprows=1)
        mileage = data[:, 0]
        price = data[:, 1]
    except FileNotFoundError:
        print("Dataset file not found. Please provide 'data.csv'.")
        return
    # Load precomputed theta values and scaling parameters
    try:
        with open("model.json", "r") as file:
            model = json.load(file)
            theta0 = model["theta0"]
            theta1 = model["theta1"]
            mileage_mean = model["mileage_mean"]
            mileage_std = model["mileage_std"]
    except FileNotFoundError:
        print("Model file not found. Please run the training program first.")
        return

    price_pred = estimate_price(mileage, theta0, theta1, mileage_mean, mileage_std)

    mse = mean_squared_error(price, price_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(price, price_pred)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
if __name__ == "__main__":
    main()


