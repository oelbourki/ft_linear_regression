import json

def estimate_price(mileage, theta0, theta1):
    """Calculate the estimated price based on the given mileage."""
    return theta0 + theta1 * mileage

def main():
    # Load precomputed theta values
    try:
        with open("model.json", "r") as file:
            model = json.load(file)
            theta0 = model["theta0"]
            theta1 = model["theta1"]
    except FileNotFoundError:
        print("Model file not found. Please run the training program first.")
        return

    # Get mileage input
    mileage = float(input("Enter the mileage of the car: "))

    # Predict and display the price
    price = estimate_price(mileage, theta0, theta1)
    print(f"The estimated price for a car with mileage {mileage} is: {price:.2f}")

if __name__ == "__main__":
    main()
