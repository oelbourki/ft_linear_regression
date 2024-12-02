import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from my_linear_regression import MyLinearRegression  # Assuming your class is in a separate file
from sklearn.metrics import mean_squared_error, r2_score
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

# Train the model
model.fit_(x_scaled, y_scaled, alpha=1.6e-4, n_cycle=100000)

# Save the trained parameters
np.save('theta.npy', model.theta)

# Save the scalers for the prediction program
with open('scalers.pkl', 'wb') as f:
    pickle.dump((scaler_x, scaler_y), f)

# Evaluate the model
y_pred_scaled = model.predict_(x_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_scaled)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

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
