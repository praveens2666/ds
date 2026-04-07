from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("students_dataset.csv")
X=df[["Marks"]]
m=KMeans(n_clusters=2, random_state=42,n_init=10)
labels=m.fit_predict(X)
plt.scatter(X['Marks'],[0]*len(X),c=labels)
plt.show()
print(m.inertia_)
print(silhouette_score(X,labels))

mark_value = float(input("Enter Marks value: "))
new_point = pd.DataFrame([[mark_value]], columns=["Marks"])
cluster_id = int(m.predict(new_point)[0])
print("Predicted Cluster:", cluster_id)














import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
df=pd.read_csv("students_dataset.csv")
x=df[["Hours_Studied"]]
y=df[["Marks"]]
xtrain,xtest,ytrain,ytest =train_test_split(x,y,train_size=0.8,random_state=42)
model=LinearRegression().fit(x,y)
pred=model.predict(xtest)
totalpred=model.predict(x)
plt.scatter(x,y)
plt.title("Actual")
plt.show()
plt.scatter(x,totalpred)
plt.title("Predicted")
plt.show()
hrs=float(input("Enter Hours Studied: "))
d=pd.DataFrame([[hrs]],columns=["Hours_Studied"])
print(model.predict(d)[0][0])
print(r2_score(ytest,pred))
print(mean_absolute_error(ytest,pred))
print(mean_squared_error(ytest,pred))







from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# df = your dataset
X = df[['study_hours_per_day']]   # input
y = df['GPA']               # output

model = LinearRegression()
model.fit(X, y)
# Predictions
y_pred = model.predict(X)

# R2 Score
r2 = r2_score(y, y_pred)

print("R2 Score:", r2)

prediction = model.predict([[5.9]])

print("Predicted value:", prediction)






import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("dataset.csv")

# -------------------------------
# PREPROCESSING
# -------------------------------

# Handle missing values (numeric only)
df = df.fillna(df.mean(numeric_only=True))

# Keep only numerical columns
df_num = df.select_dtypes(include='number')

# -------------------------------
# SCALING (VERY IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_num)
# -------------------------------
# APPLY K-MEANS (choose K=3 or from elbow)
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

print("Cluster Labels:\n", labels[:10])

# -------------------------------
# VISUALIZATION
# -------------------------------
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()









#Classification Section

#================================================XGBOOST MODEL================================
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Metrics
print("XGBoost Results:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
