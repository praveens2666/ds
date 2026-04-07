import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("student.csv")


X = df.drop("GPA", axis=1)
y = df["GPA"]

# -------------------------------
# 2. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 3. SCALING
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. MODELS
# -------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

# -------------------------------
# 5. TRAIN & EVALUATE
# -------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n======", name, "======")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 6. POLYNOMIAL REGRESSION
# -------------------------------
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

print("\n====== Polynomial Regression ======")
print("MAE:", mean_absolute_error(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("R2 Score:", r2_score(y_test, y_pred_poly))

# -------------------------------
# 7. SINGLE PREDICTION (SAFE WAY)
# -------------------------------
sample = X_test[0:1]

for name, model in models.items():
    pred = model.predict(sample)
    print(f"{name} Prediction:", pred[0])

# Polynomial prediction
sample_poly = poly.transform(sample)
prediction_poly = poly_model.predict(sample_poly)

# print("\nSample Prediction (Normal):", prediction[0])
print("Sample Prediction (Polynomial):", prediction_poly[0])