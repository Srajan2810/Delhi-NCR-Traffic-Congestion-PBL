
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import linprog

# Create results folder
os.makedirs("results", exist_ok=True)

# Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

# --------------------------------
# 1. DESCRIPTIVE STATISTICS
# --------------------------------

numeric_columns = [
    "Traffic_Volume_vehicles_per_interval",
    "Average_Speed_kmph",
    "Queue_Length_vehicles",
    "Signal_Wait_Time_seconds",
    "Estimated_Travel_Time_minutes"
]

statistics = df[numeric_columns].describe().T
statistics["Variance"] = df[numeric_columns].var()

# --------------------------------
# 2. CORRELATION ANALYSIS
# --------------------------------

correlation = df[numeric_columns].corr()

# --------------------------------
# 3. LINEAR REGRESSION
# --------------------------------

features = [
    "Traffic_Volume_vehicles_per_interval",
    "Average_Speed_kmph",
    "Queue_Length_vehicles",
    "Signal_Wait_Time_seconds",
    "Observed_Distance_km_illustrative"
]

X = df[features]
y = df["Estimated_Travel_Time_minutes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

regression_results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
    "Value": [
        mean_absolute_error(y_test, predictions),
        mean_squared_error(y_test, predictions),
        np.sqrt(mean_squared_error(y_test, predictions)),
        r2_score(y_test, predictions)
    ]
})

# --------------------------------
# 4. GRADIENT DESCENT
# --------------------------------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

m, n = X_train_scaled.shape
weights = np.zeros(n)
bias = 0
learning_rate = 0.05
iterations = 5000

for i in range(iterations):
    predicted = np.dot(X_train_scaled, weights) + bias
    errors = predicted - y_train.values

    dw = np.dot(X_train_scaled.T, errors) / m
    db = np.sum(errors) / m

    weights -= learning_rate * dw
    bias -= learning_rate * db

gd_predictions = np.dot(X_test_scaled, weights) + bias

gradient_results = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R-squared"],
    "Value": [
        mean_absolute_error(y_test, gd_predictions),
        mean_squared_error(y_test, gd_predictions),
        np.sqrt(mean_squared_error(y_test, gd_predictions)),
        r2_score(y_test, gd_predictions)
    ]
})

# --------------------------------
# 5. LINEAR PROGRAMMING
# --------------------------------

lp = linprog(
    [-8, -5],
    A_eq=[[1, 1]],
    b_eq=[120],
    bounds=[(30, 90), (30, 90)],
    method="highs"
)

lp_results = pd.DataFrame({
    "Signal Phase": ["North-South", "East-West"],
    "Optimized Green Time (seconds)": lp.x if lp.success else [np.nan, np.nan]
})

# --------------------------------
# 6. EXPORT ALL RESULTS
# --------------------------------

output_file = "results/Complete_Project_Results.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    statistics.to_excel(writer, sheet_name="Descriptive Statistics")
    correlation.to_excel(writer, sheet_name="Correlation")
    regression_results.to_excel(writer, sheet_name="Regression", index=False)
    gradient_results.to_excel(writer, sheet_name="Gradient Descent", index=False)
    lp_results.to_excel(writer, sheet_name="Linear Programming", index=False)

print("\n===== PROJECT SUMMARY GENERATED =====")

print("Descriptive statistics completed.")
print("Correlation analysis completed.")
print("Regression evaluation completed.")
print("Gradient Descent evaluation completed.")
print("Linear Programming completed.")

print("\nSaved at:", output_file)
