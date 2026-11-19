
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

# Create folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Select numerical variables
columns = [
    "Traffic_Volume_vehicles_per_interval",
    "Average_Speed_kmph",
    "Queue_Length_vehicles",
    "Signal_Wait_Time_seconds",
    "Estimated_Travel_Time_minutes"
]

data = df[columns]

# Calculate Pearson correlation
correlation_matrix = data.corr(method="pearson")

print("\n===== PEARSON CORRELATION MATRIX =====")
print(correlation_matrix.round(3))

# Save correlation values
correlation_matrix.to_csv(
    "results/correlation_matrix.csv"
)

# Generate heatmap
plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Pearson Correlation Matrix - Traffic Dataset")
plt.tight_layout()

plt.savefig("graphs/06_correlation_heatmap.png")
plt.show()

# Scatter plot: Traffic Volume vs Travel Time
plt.figure(figsize=(8, 6))

sns.regplot(
    data=df,
    x="Traffic_Volume_vehicles_per_interval",
    y="Estimated_Travel_Time_minutes"
)

plt.title("Traffic Volume vs Estimated Travel Time")
plt.xlabel("Traffic Volume")
plt.ylabel("Estimated Travel Time (minutes)")
plt.tight_layout()

plt.savefig("graphs/07_traffic_vs_travel_time.png")
plt.show()

print("\nCorrelation analysis completed!")

