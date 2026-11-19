
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("data/Delhi_NCR_Traffic_Dataset_100_Records.csv")

# Create graphs folder if it doesn't exist
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# Set time period order
time_order = [
    "Morning Peak",
    "Late Morning",
    "Afternoon",
    "Evening Peak",
    "Night"
]

# ---------------------------------
# GRAPH 1: Traffic Volume by Time
# ---------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Time_Period",
    y="Traffic_Volume_vehicles_per_interval",
    order=time_order,
    errorbar=None
)

plt.title("Average Traffic Volume by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Average Traffic Volume")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("graphs/01_traffic_by_time.png")
plt.show()


# ---------------------------------
# GRAPH 2: Average Speed by Location
# ---------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Location",
    y="Average_Speed_kmph",
    errorbar=None
)

plt.title("Average Vehicle Speed by Location")
plt.xlabel("Location")
plt.ylabel("Average Speed (km/h)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("graphs/02_speed_by_location.png")
plt.show()


# ---------------------------------
# GRAPH 3: Travel Time by Time Period
# ---------------------------------

plt.figure(figsize=(10, 6))

sns.barplot(
    data=df,
    x="Time_Period",
    y="Estimated_Travel_Time_minutes",
    order=time_order,
    errorbar=None
)

plt.title("Average Travel Time by Time Period")
plt.xlabel("Time Period")
plt.ylabel("Travel Time (minutes)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("graphs/03_travel_time_by_period.png")
plt.show()


# ---------------------------------
# GRAPH 4: Traffic Volume Distribution
# ---------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Traffic_Volume_vehicles_per_interval"],
    bins=15,
    kde=True
)

plt.title("Distribution of Traffic Volume")
plt.xlabel("Traffic Volume")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("graphs/04_traffic_distribution.png")
plt.show()


# ---------------------------------
# GRAPH 5: Traffic Volume by Location
# ---------------------------------

plt.figure(figsize=(11, 6))

sns.boxplot(
    data=df,
    x="Location",
    y="Traffic_Volume_vehicles_per_interval"
)

plt.title("Traffic Volume Comparison Across Locations")
plt.xlabel("Location")
plt.ylabel("Traffic Volume")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig("graphs/05_location_comparison.png")
plt.show()


print("All 5 graphs generated successfully!")

