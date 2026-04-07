import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Features & target
X = df[['study_hours_per_day']]
y = df['GPA']

# -------------------------------
# 1. SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 2. TRAIN MODEL
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 3. PREDICT ON TEST DATA
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 4. EVALUATION
# -------------------------------
r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

# -------------------------------
# 5. SINGLE PREDICTION
# -------------------------------
prediction = model.predict([[5.9]])
print("Predicted GPA:", prediction[0])