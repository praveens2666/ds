import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# -------------------------------
# 1. LOAD PREPROCESSED DATA
# -------------------------------
df = pd.read_excel("intent_aware_transaction_dataset.xlsx")

# -------------------------------
# 2. SPLIT FEATURES & TARGET
# -------------------------------
bool_cols = df.sleecct_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
for col in df.columns:
    if df[col].nunique() == len(df):
        df.drop(columns=[col],inplace=True)

X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------
# 3. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. MODELS
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# -------------------------------
# 5. TRAIN & EVALUATE
# -------------------------------
for name, model in models.items():

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n======", name, "======")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

# -------------------------------
# 6. SINGLE PREDICTION (SAFE WAY)
# -------------------------------
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

print("\nSample Prediction:", prediction[0])