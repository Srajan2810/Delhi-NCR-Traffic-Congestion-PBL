
import pandas as pd
import os

# 1. Define the dataset path
file_path = "data/Delhi_NCR_Traffic_Dataset_100_Records.csv"

# 2. Load the CSV file
df = pd.read_csv(file_path)

# 3. Display the first 5 records
print("\n----- FIRST 5 RECORDS -----")
print(df.head())

# 4. Display dataset dimensions
print("\n----- DATASET SHAPE -----")
print("Rows and Columns:", df.shape)

# 5. Display column names
print("\n----- COLUMN NAMES -----")
print(df.columns.tolist())

# 6. Check data types
print("\n----- DATA TYPES -----")
print(df.dtypes)

# 7. Check missing values
print("\n----- MISSING VALUES -----")
print(df.isnull().sum())

# 8. Generate statistical summary
print("\n----- STATISTICAL SUMMARY -----")
print(df.describe())

# 9. Check unique locations
print("\n----- LOCATIONS -----")
print(df["Location"].unique())

# 10. Check time periods
print("\n----- TIME PERIODS -----")
print(df["Time_Period"].unique())

print("\nDataset loaded successfully!")

