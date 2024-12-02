import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from my_linear_regression import MyLinearRegression  # Assuming your class is in a separate file
import matplotlib.pyplot as plt
import pickle

# Load the data
data = pd.read_csv('data.csv')

# Extract features (mileage) and target (price)
x = data['km'].values.reshape(-1, 1)
y = data['price'].values.reshape(-1, 1)

# Normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

# Initialize theta with small random values
theta = np.random.randn(2, 1) * np.sqrt(2. / x_scaled.shape[0])
model = MyLinearRegression(theta)

# Train the model and store theta values at periodic intervals
model.fit_(x_scaled, y_scaled, alpha=1.6e-4, n_cycle=100000, save_interval=5000)

# Save the trained parameters
np.save('theta.npy', model.theta)

# Save the scalers for the prediction program
with open('scalers.pkl', 'wb') as f:
    pickle.dump((scaler_x, scaler_y), f)

# Evaluate the model
y_pred_scaled = model.predict_(x_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_scaled)

# Metrics
mse = model.mse_(x_scaled, y_scaled)
rmse = model.rmse_(x_scaled, y_scaled)
r2 = model.r2score_(x_scaled, y_scaled)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot multiple regression lines during training

# Convert history to numpy array for easy access
theta_history = np.array(model.history).squeeze()  # Convert list to numpy array for easier indexing

# Set up a plot for multiple regression lines
plt.figure(figsize=(10, 6))

# Loop over stored theta values and plot the corresponding regression line
for epoch, theta_at_epoch in enumerate(theta_history):
    # Extract intercept (theta[0]) and slope (theta[1])
    intercept = theta_at_epoch[0]
    slope = theta_at_epoch[1]
    
    # Compute predictions with the current theta
    y_pred_epoch = intercept + slope * x_scaled
    
    # Plot the regression line for this epoch
    plt.plot(x, scaler_y.inverse_transform(y_pred_epoch), label=f'Epoch {epoch * 5000}', alpha=0.7)

# Plot the final regression line
plt.plot(x, y_pred, label='Final Regression Line', color='red', linewidth=2)

# Plot the true values as scatter points
plt.scatter(x, y, label='True Values', color='blue')

# Labeling the plot
plt.xlabel('Mileage (km)')
plt.ylabel('Price')
plt.legend()
plt.title('Linear Regression: Evolution of Regression Line During Training')
plt.show()
