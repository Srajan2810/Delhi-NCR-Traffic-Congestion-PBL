
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. Select input and output variables
X = df[
    [
        "Traffic_Volume_vehicles_per_interval",
        "Average_Speed_kmph",
        "Queue_Length_vehicles",
        "Signal_Wait_Time_seconds",
        "Observed_Distance_km_illustrative"
    ]
].values

y = df["Estimated_Travel_Time_minutes"].values

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize input features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Initialize parameters
m, n = X_train.shape

weights = np.zeros(n)
bias = 0

learning_rate = 0.05
iterations = 5000

cost_history = []

# 6. Gradient Descent
for i in range(iterations):

    # Calculate predictions
    predictions = np.dot(X_train, weights) + bias

    # Calculate errors
    errors = predictions - y_train

    # Calculate gradients
    dw = (1 / m) * np.dot(X_train.T, errors)
    db = (1 / m) * np.sum(errors)

    # Update parameters
    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    # Calculate cost
    cost = np.sum(errors ** 2) / (2 * m)
    cost_history.append(cost)

# 7. Predict using optimized parameters
y_pred = np.dot(X_test, weights) + bias

# 8. Evaluate model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== GRADIENT DESCENT RESULTS =====")

print("Optimized weights:", weights)
print("Optimized bias:", round(bias, 4))

print("MSE:", round(mse, 4))
print("RMSE:", round(rmse, 4))
print("R-squared:", round(r2, 4))

print("\nInitial Cost:", round(cost_history[0], 4))
print("Final Cost:", round(cost_history[-1], 4))

# 9. Plot cost reduction
plt.figure(figsize=(9, 6))

plt.plot(cost_history)

plt.title("Gradient Descent - Cost Reduction")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE / 2)")

plt.tight_layout()
plt.savefig("graphs/09_gradient_descent_cost.png")
plt.show()

# 10. Save predictions
pd.DataFrame({
    "Actual_Travel_Time": y_test,
    "Predicted_Travel_Time": y_pred
}).to_csv(
    "results/gradient_descent_predictions.csv",
    index=False
)

print("\nGradient Descent completed successfully!")

