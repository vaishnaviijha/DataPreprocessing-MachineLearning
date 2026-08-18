import pandas as pd

# 1. Column names (the raw .data file has no header row)
columns = [
    "symboling", "normalized_losses", "make", "fuel_type", "aspiration",
    "num_of_doors", "body_style", "drive_wheels", "engine_location",
    "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
    "num_of_cylinders", "engine_size", "fuel_system", "bore", "stroke",
    "compression_ratio", "horsepower", "peak_rpm", "city_mpg",
    "highway_mpg", "price",
]

# 2. Load the data — treat "?" as missing right away
df = pd.read_csv(r"C:\Users\VAISHNAVI KUMARI\Documents\SEM3\AppliedMachineLearning\Datasets_Ppts\EXP1\dataset_1.data", names=columns, na_values="?")

# 3. Quick look at the data
print(df.shape)        
print(df.head())
print(df.tail())

# 4. Data types + missing values in one table
info = pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isnull().sum()
})
print(info)

# 5. Summary statistics
print(df.describe())               
print(df.describe(include="all"))  

# 6. Save the cleaned dataset
df.to_csv("automobile_clean.csv", index=False)