import pandas as pd

# Load CSV
df = pd.read_csv(r"C:\flipkart-gridlock\Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv")

# Basic inspection
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nSummary statistics:")
print(df.describe(include="all"))