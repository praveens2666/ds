import numpy as np
import pandas as pd
from scipy import stats

# -------------------------------
# Read CSV
# -------------------------------
df = pd.read_excel("intent_aware_transaction_dataset.xlsx")  # use your file

# -------------------------------
# Select only numerical columns
# -------------------------------
num_df = df.select_dtypes(include=np.number)

print("Numerical Columns:", num_df.columns.tolist())

# -------------------------------
# Column-wise analysis
# -------------------------------
for col in num_df.columns:
    print(f"\n===== Column: {col} =====")

    column_data = num_df[col].dropna()

    # Central Tendency
    mean = column_data.mean()
    median = column_data.median()
    mode = column_data.mode()[0] if not column_data.mode().empty else "No mode"

    print("\nCentral Tendency:")
    print("Mean :", mean)
    print("Median :", median)
    print("Mode :", mode)

    # Dispersion
    range_val = column_data.max() - column_data.min()
    variance = column_data.var()
    std_dev = column_data.std()
    iqr = column_data.quantile(0.75) - column_data.quantile(0.25)

    print("\nDispersion:")
    print("Range :", range_val)
    print("Variance :", variance)
    print("Std Dev :", std_dev)
    print("IQR :", iqr)

    # Shape
    skewness = stats.skew(column_data)
    kurtosis = stats.kurtosis(column_data)

    print("\nShape:")
    print("Skewness :", skewness)
    print("Kurtosis :", kurtosis)

    # Interpretation
    print("\nInterpretation:")

    if skewness > 0:
        print("Positively Skewed")
    elif skewness < 0:
        print("Negatively Skewed")
    else:
        print("Symmetrical")

    if kurtosis > 0:
        print("Leptokurtic")
    elif kurtosis < 0:
        print("Platykurtic")
    else:
        print("Mesokurtic")