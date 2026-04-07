# =========================================
# COMPLETE DATA SCIENCE PIPELINE (FINAL)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_excel("intent_aware_transaction_dataset.xlsx")

print("Initial Shape:", df.shape)
print(df.head())

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# -------------------------------
# 2. CHECK DATA (VERY IMPORTANT)
# -------------------------------

print("\n Missing Values:")
print(df.isnull().sum())

if "label" in df.columns:
    print("\n Class Distribution:")
    print(df["label"].value_counts())

print("\n Feature Ranges:")
for col in df.select_dtypes(include=np.number).columns:
    print(col, "-> Min:", df[col].min(), "Max:", df[col].max())

# -------------------------------
# 3. DROP USELESS COLUMNS
# -------------------------------
for col in df.columns:
    if df[col].nunique() == len(df):
        df.drop(columns=[col], inplace=True)

# -------------------------------
# 4. SPLIT FEATURES & TARGET
# -------------------------------
target_col = "label" if "label" in df.columns else df.columns[-1]

X = df.drop(target_col, axis=1)
y = df[target_col]

# -------------------------------
# 5. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. HANDLE MISSING VALUES
# -------------------------------
num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(exclude=np.number).columns

# Numeric → Median
num_imputer = SimpleImputer(strategy="median")
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Categorical → Mode
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

# -------------------------------
# 7. ENCODING
# -------------------------------
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

# -------------------------------
# 8. HANDLE IMBALANCE
# -------------------------------
if y_train.value_counts().max() > 2 * y_train.value_counts().min():
    over = RandomOverSampler(random_state=42)
    X_train, y_train = over.fit_resample(X_train, y_train)
    print("Class imbalance handled")

# -------------------------------
# 9. NORMALIZATION (SCALING)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Normalized")

# -------------------------------
# 10. FINAL DATASET
# -------------------------------
final_df = pd.DataFrame(X_train)
final_df["label"] = y_train.values

print("Final Shape:", final_df.shape)

# Save
final_df.to_csv("final_preprocessed_data.csv", index=False)
print("Saved Successfully")

# =========================================
# 11. VISUALIZATION
# =========================================

df = final_df

# Histogram
df.hist(figsize=(10, 8))
plt.suptitle("Histogram")
plt.show()

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df)
plt.title("Boxplot")
plt.xticks(rotation=45)
plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Scatter (auto pick first 2 cols)
cols = df.columns
if len(cols) >= 2:
    plt.scatter(df[cols[0]], df[cols[1]])
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title("Scatter Plot")
    plt.show()

# Pairplot (IMPORTANT)
df_sample = df.sample(50)  # only 1000 rows
sns.pairplot(df_sample)
plt.show()
# sns.pairplot(df)
# plt.show()