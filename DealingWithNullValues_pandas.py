import pandas as pd

# 1. Column names + load data (treat "?" as missing right away)
headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg",
    "highway-mpg", "price",
]

df = pd.read_csv(r"C:\Users\VAISHNAVI KUMARI\Documents\SEM3\AppliedMachineLearning\Datasets_Ppts\EXP2\car_dataset.data", names=headers, na_values="?")

# 2. See how many missing values are in each column
print("Missing values before cleaning:\n")
print(df.isnull().sum())

# 3. Fill numeric columns' missing values with their column mean
mean_fill_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]
for col in mean_fill_cols:
    avg = df[col].astype("float").mean()
    df[col] = df[col].fillna(avg)

# 4. Fill "num-of-doors" (categorical) with its most frequent value
most_common_doors = df["num-of-doors"].value_counts().idxmax()
df["num-of-doors"] = df["num-of-doors"].fillna(most_common_doors)

# 5. Drop rows where "price" is missing (can't use a row with no target value)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# 6. Confirm everything is clean now
print("\nMissing values after cleaning:\n")
print(df.isnull().sum())
print("\nFinal shape:", df.shape)

# 7. Save the cleaned dataset
df.to_csv("car_dataset_no_missing.csv", index=False)
