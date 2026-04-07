import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("final_preprocessed_data.csv")

# -------------------------------
# 2. SPLIT FEATURES & LABEL
# -------------------------------
X = df.drop("label", axis=1)
y_true = df["label"]

# -------------------------------
# 3. SCALING (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 4. K-MEANS
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
k_labels = kmeans.fit_predict(X_scaled)

print("\n=== K-Means ===")
print("Silhouette:", silhouette_score(X_scaled, k_labels))
print("Davies:", davies_bouldin_score(X_scaled, k_labels))
print("Calinski:", calinski_harabasz_score(X_scaled, k_labels))

# -------------------------------
# 5. AGGLOMERATIVE
# -------------------------------
agglo = AgglomerativeClustering(n_clusters=3)
a_labels = agglo.fit_predict(X_scaled)

print("\n=== Agglomerative ===")
print("Silhouette:", silhouette_score(X_scaled, a_labels))
print("Davies:", davies_bouldin_score(X_scaled, a_labels))
print("Calinski:", calinski_harabasz_score(X_scaled, a_labels))

# -------------------------------
# 6. VISUALIZATION (PCA)
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=k_labels)
plt.title("K-Means Clustering")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()