import json

def estimate_price(mileage, theta0, theta1, mileage_mean, mileage_std):
    """Calculate the estimated price based on the given mileage."""
    # Apply the same scaling used during training
    mileage_normalized = (mileage - mileage_mean) / mileage_std
    return theta0 + theta1 * mileage_normalized

def main():
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

    # Get mileage input
    try:
        mileage = float(input("Enter the mileage of the car: "))
    except ValueError:
        print("Invalid input. Please enter a numeric value.")
        return

    # Predict and display the price
    price = estimate_price(mileage, theta0, theta1, mileage_mean, mileage_std)
    print(f"The estimated price for a car with mileage {mileage} is: {price:.2f}")

if __name__ == "__main__":
    main()
