import kagglehub
import pandas as pd
import os

# Download latest version
path = kagglehub.dataset_download("tanuprabhu/linear-regression-dataset")

print("Path to dataset files:", path)

# Load the dataset
dataset_file = os.path.join(path, "Linear Regression - Sheet1.csv")
df = pd.read_csv(dataset_file)

# Display basic information
print("\n=== Dataset Information ===")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset statistics:")
print(df.describe())
print(f"\nMissing values:")
print(df.isnull().sum())
