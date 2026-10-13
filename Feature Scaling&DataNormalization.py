import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv(r"C:\Users\VAISHNAVI KUMARI\Documents\SEM3\AppliedMachineLearning\Datasets_Ppts\EXP6\Dataset_03.csv")
print(df.head())

# 2. Split into features (X) and target (y)
# X = every column except the last, y = the last column ("Purchased")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. Split into train/test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# 4. Scale only the numeric columns (Salary, YearsExperience — columns 4 onward)
# Convert to float first, since scaled values won't fit back into an int column
X_train = X_train.astype({col: "float64" for col in X_train.columns[4:]})
X_test = X_test.astype({col: "float64" for col in X_test.columns[4:]})

# fit_transform on the training set: learns mean/std AND scales it in one step
# transform only on the test set: reuses the training mean/std, avoids data leakage
sc = StandardScaler()
X_train.iloc[:, 4:] = sc.fit_transform(X_train.iloc[:, 4:])
X_test.iloc[:, 4:] = sc.transform(X_test.iloc[:, 4:])

# 5. Check the results
print("\nScaled X_train:\n", X_train)
print("\nScaled X_test:\n", X_test)