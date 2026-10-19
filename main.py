import pandas as pd

# Load dataset
df = pd.read_csv("Final_data.csv")

print("ORIGINAL DATASET")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# Remove unnecessary index column
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove duplicate records
df = df.drop_duplicates()

# Convert numeric columns safely
numeric_cols = ["Cases", "Deaths", "day", "mon", "year",
                "Latitude", "Longitude", "preci", "LAI", "Temp"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Check missing values before cleaning
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Keep missing death counts as unknown
# Fill missing climate values with their column median
for col in ["preci", "LAI", "Temp"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

# Save cleaned dataset
df.to_csv("cleaned_health_data.csv", index=False)

print("\nCLEANING COMPLETED")
print("Rows after cleaning:", len(df))
print("Columns after cleaning:", len(df.columns))
print("\nRemaining missing values:")
print(df.isnull().sum())
print("\nSaved as cleaned_health_data.csv")
