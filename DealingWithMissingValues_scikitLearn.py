import pandas as pd
from sklearn.impute import SimpleImputer

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

# 3. Fill numeric columns' missing values with their column mean, using sklearn
mean_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]
mean_imputer = SimpleImputer(strategy="mean")
df[mean_cols] = mean_imputer.fit_transform(df[mean_cols])

# 4. Fill "num-of-doors" (categorical) with its most frequent value, using sklearn
mode_imputer = SimpleImputer(strategy="most_frequent")
df[["num-of-doors"]] = mode_imputer.fit_transform(df[["num-of-doors"]])

# 5. Drop rows where "price" is missing (can't use a row with no target value)
df.dropna(subset=["price"], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)

# 6. Confirm everything is clean now
print("\nMissing values after cleaning:\n")
print(df.isnull().sum())
print("\nFinal shape:", df.shape)

# 7. Save the cleaned dataset
df.to_csv("car_dataset_no_missing_sklearn.csv", index=False)
