
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Load dataset
df = pd.read_csv(
    "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
)

os.makedirs("graphs", exist_ok=True)

traffic = df[
    "Traffic_Volume_vehicles_per_interval"
].values

# CLT experiment
sample_size = 30
number_of_samples = 1000

sample_means = []

np.random.seed(42)

# Draw repeated samples with replacement
for i in range(number_of_samples):

    sample = np.random.choice(
        traffic,
        size=sample_size,
        replace=True
    )

    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

# Calculate values
original_mean = np.mean(traffic)
mean_of_means = np.mean(sample_means)

standard_error = np.std(sample_means, ddof=1)

print("\n===== CENTRAL LIMIT THEOREM =====")

print("Original Dataset Mean:", round(original_mean, 2))
print("Mean of Sample Means:", round(mean_of_means, 2))
print("Standard Error of Sample Means:", round(standard_error, 2))

# Plot sampling distribution
plt.figure(figsize=(9, 6))

plt.hist(
    sample_means,
    bins=25,
    density=True,
    edgecolor="black"
)

plt.axvline(
    original_mean,
    linestyle="--",
    label="Original Dataset Mean"
)

plt.title("Central Limit Theorem - Sampling Distribution")
plt.xlabel("Sample Mean Traffic Volume")
plt.ylabel("Density")
plt.legend()

plt.tight_layout()

plt.savefig("graphs/12_central_limit_theorem.png")
plt.show()

print("\nCLT demonstration completed!")

