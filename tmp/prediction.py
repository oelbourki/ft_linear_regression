import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

# Load trained parameters and scalers
theta = np.load('theta.npy')
with open('scalers.pkl', 'rb') as f:
    scaler_x, scaler_y = pickle.load(f)

def predict_price(mileage, theta, scaler_x, scaler_y):
    """
    Predict the price based on mileage using the trained model.
    Args:
        mileage (float): The mileage of the car in kilometers.
        theta (np.ndarray): Model parameters.
        scaler_x (MinMaxScaler): Scaler for mileage (input feature).
        scaler_y (MinMaxScaler): Scaler for price (target variable).
    Returns:
        float: Predicted price of the car.
    """
    mileage_scaled = scaler_x.transform(np.array([[mileage]]))
    mileage_scaled = np.c_[np.ones((mileage_scaled.shape[0], 1)), mileage_scaled]
    price_scaled = mileage_scaled.dot(theta)
    price = scaler_y.inverse_transform(price_scaled)
    return price[0][0]

def validate_input(user_input):
    """
    Validate user input to ensure it's a positive number.
    Args:
        user_input (str): User input as a string.
    Returns:
        float: Validated mileage as a float.
        None: If the input is invalid.
    """
    try:
        mileage = float(user_input)
        if mileage < 0:
            print("Mileage must be a positive number. Please try again.")
            return None
        return mileage
    except ValueError:
        print("Invalid input. Please enter a numeric value for mileage.")
        return None

# Main program loop
while True:
    user_input = input("Enter the mileage of the car (in km) or type 'exit' to quit: ").strip()
    if user_input.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break

    mileage = validate_input(user_input)
    if mileage is not None:
        predicted_price = predict_price(mileage, theta, scaler_x, scaler_y)
        print(f"Estimated price for a car with {mileage} km: {predicted_price:.2f}")
