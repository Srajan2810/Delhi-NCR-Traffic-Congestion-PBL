
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy.stats import norm

# Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Select traffic volume
traffic = df["Traffic_Volume_vehicles_per_interval"]

# Calculate mean and standard deviation
mean = traffic.mean()
std = traffic.std()

print("\n===== PROBABILITY DISTRIBUTION =====")

print("Mean Traffic Volume:", round(mean, 2))
print("Standard Deviation:", round(std, 2))

# Calculate probability of traffic exceeding 1000
threshold = 1000

probability = 1 - norm.cdf(
    threshold,
    loc=mean,
    scale=std
)

print(
    "Model-based probability of traffic exceeding",
    threshold,
    ":",
    round(probability * 100, 2),
    "%"
)

# Plot histogram and fitted normal distribution
plt.figure(figsize=(9, 6))

sns.histplot(
    traffic,
    bins=15,
    stat="density",
    kde=False,
    label="Traffic Volume"
)

x = np.linspace(traffic.min(), traffic.max(), 300)

plt.plot(
    x,
    norm.pdf(x, mean, std),
    linewidth=2,
    label="Fitted Normal Distribution"
)

plt.title("Traffic Volume - Normal Distribution Model")
plt.xlabel("Traffic Volume")
plt.ylabel("Probability Density")
plt.legend()

plt.tight_layout()
plt.savefig("graphs/11_probability_distribution.png")
plt.show()

# Save summary
with open("results/probability_results.txt", "w") as file:
    file.write(f"Mean: {mean:.2f}\n")
    file.write(f"Standard Deviation: {std:.2f}\n")
    file.write(f"Model-based probability above 1000: {probability:.4f}\n")

print("\nProbability analysis completed!")

