
import pandas as pd

# Load dataset
file_path = "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"
df = pd.read_csv(file_path)

# Select numerical columns
columns = [
    "Traffic_Volume_vehicles_per_interval",
    "Average_Speed_kmph",
    "Queue_Length_vehicles",
    "Signal_Wait_Time_seconds",
    "Estimated_Travel_Time_minutes"
]

print("\n===== DESCRIPTIVE STATISTICAL ANALYSIS =====")

for col in columns:

    print("\nVariable:", col)

    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    variance = df[col].var()
    std_dev = df[col].std()
    data_range = df[col].max() - df[col].min()

    print("Mean:", round(mean, 2))
    print("Median:", round(median, 2))
    print("Mode:", round(mode, 2))
    print("Variance:", round(variance, 2))
    print("Standard Deviation:", round(std_dev, 2))
    print("Range:", round(data_range, 2))

print("\nStatistical analysis completed!")
