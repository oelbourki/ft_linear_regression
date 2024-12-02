import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression  # Assuming your class is in a separate file
import matplotlib.pyplot as plt
import pickle

# Load the data
data = pd.read_csv('data.csv')

# Extract features (mileage) and target (price)
x = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

# Normalize the data
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()

x_scaled = (x - x_min) / (x_max - x_min)
y_scaled = (y - y_min) / (y_max - y_min)

# Initialize theta with small random values
theta = np.random.randn(2, 1) * np.sqrt(2. / x_scaled.shape[0])
model = MyLinearRegression(theta)

# Train the model
model.fit_(x_scaled, y_scaled, alpha=1.6e-4, n_cycle=100000)

# Save the trained parameters
np.save('theta.npy', model.theta)

# Save the normalization parameters for the prediction program
scalers = {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}
with open('scalers.pkl', 'wb') as f:
    pickle.dump(scalers, f)

# Evaluate the model
y_pred_scaled = model.predict_(x_scaled)

# Denormalize predictions and true values
y_pred = y_pred_scaled * (y_max - y_min) + y_min
y_true = y_scaled * (y_max - y_min) + y_min

# Metrics calculated using the class
mse = model.mse_(x_scaled, y_scaled)
rmse = model.rmse_(x_scaled, y_scaled)
r2 = model.r2score_(x_scaled, y_scaled)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.scatter(x, y, label='True Values', color='blue')
plt.plot(x, y_pred, label='Regression Line', color='red')
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression: Mileage vs Price')
plt.show()
