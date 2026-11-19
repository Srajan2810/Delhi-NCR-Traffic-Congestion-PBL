
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1. Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 2. Select independent variables (X)
X = df[
    [
        "Traffic_Volume_vehicles_per_interval",
        "Average_Speed_kmph",
        "Queue_Length_vehicles",
        "Signal_Wait_Time_seconds",
        "Observed_Distance_km_illustrative"
    ]
]

# 3. Select dependent variable (Y)
Y = df["Estimated_Travel_Time_minutes"]

# 4. Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 5. Create and train regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# 6. Predict travel time
Y_pred = model.predict(X_test)

# 7. Display regression coefficients
print("\n===== REGRESSION EQUATION =====")

print("Intercept:", round(model.intercept_, 4))

for name, coefficient in zip(X.columns, model.coef_):
    print(name, ":", round(coefficient, 4))

# 8. Evaluate model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("\n===== MODEL PERFORMANCE =====")
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R-squared:", round(r2, 4))

# 9. Save actual and predicted values
predictions = pd.DataFrame({
    "Actual_Travel_Time": Y_test.values,
    "Predicted_Travel_Time": Y_pred
})

predictions.to_csv(
    "results/regression_predictions.csv",
    index=False
)

# 10. Actual vs predicted graph
plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=Y_test,
    y=Y_pred
)

minimum = min(Y_test.min(), Y_pred.min())
maximum = max(Y_test.max(), Y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.title("Actual vs Predicted Travel Time")
plt.xlabel("Actual Travel Time (minutes)")
plt.ylabel("Predicted Travel Time (minutes)")
plt.tight_layout()

plt.savefig("graphs/08_actual_vs_predicted.png")
plt.show()

print("\nLinear Regression completed successfully!")

