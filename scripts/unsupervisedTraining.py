# ===============================================================
# Phase 3: Unsupervised Model Training – Airline Delay Clustering
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pickle   # 👈 added for saving models

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
df = pd.read_csv("Airline-Analytics-using-AI-ML\data\cleaned_flights.csv")  # 👈 replace with your file path
print("Original Shape:", df.shape)

# ---------------------------------------------------------------
# 2. Select numeric features
# ---------------------------------------------------------------
numeric_df = df.select_dtypes(include=[np.number]).drop(columns=['CANCELLATION_REASON'], errors='ignore')
print("Numeric columns used:", numeric_df.columns.tolist())
print("Before imputation:", numeric_df.shape)

# ---------------------------------------------------------------
# 3. Handle missing values properly
# ---------------------------------------------------------------
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(numeric_df)
print("Any NaN left after imputation?", np.isnan(X_imputed).any())

# ---------------------------------------------------------------
# 4. Scale the data
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)
print("Shape after scaling:", X_scaled.shape)

# ---------------------------------------------------------------
# 5. (Optional) Remove zero-variance features
# ---------------------------------------------------------------
vt = VarianceThreshold(threshold=0.0)
X_scaled = vt.fit_transform(X_scaled)
print("After removing zero-variance features:", X_scaled.shape)

# ---------------------------------------------------------------
# 6. Dimensionality Reduction using PCA
# ---------------------------------------------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print("PCA shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# ---------------------------------------------------------------
# 7. K-Means Clustering
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

sil_kmeans = silhouette_score(X_pca, kmeans_labels)
print(f"\nK-Means Silhouette Score: {sil_kmeans:.4f}")
print("KMeans Cluster Centers (PCA space):")
print(kmeans.cluster_centers_)

# ---------------------------------------------------------------
# 8. DBSCAN Clustering
# ---------------------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=10)
db_labels = dbscan.fit_predict(X_pca)

# Filter out noise (-1) before silhouette score
if len(set(db_labels)) > 1 and -1 not in set(db_labels):
    sil_dbscan = silhouette_score(X_pca, db_labels)
else:
    sil_dbscan = "N/A (DBSCAN produced noise-only or single cluster)"

print(f"\nDBSCAN Silhouette Score: {sil_dbscan}")
unique_labels = set(db_labels)
print("DBSCAN clusters found:", unique_labels)

# ---------------------------------------------------------------
# 9. Combine cluster results back to original data
# ---------------------------------------------------------------
df_clusters = df.copy()
df_clusters["KMeans_Cluster"] = kmeans_labels
df_clusters["DBSCAN_Cluster"] = db_labels

# ---------------------------------------------------------------
# 10. Save results
# ---------------------------------------------------------------
df_clusters.to_csv("unsupervised_clusters_output.csv", index=False)
print("\n✅ Unsupervised clustering complete.")
print("Saved to: unsupervised_clusters_output.csv")

# ---------------------------------------------------------------
# 11. ✅ Save models and preprocessing pipeline using pickle
# ---------------------------------------------------------------
model_bundle = {
    "imputer": imputer,
    "scaler": scaler,
    "pca": pca,
    "kmeans": kmeans
}

with open("unsupervised_model_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

# (Optional) Save DBSCAN separately
with open("dbscan_model.pkl", "wb") as f:
    pickle.dump(dbscan, f)

print("\n✅ Model pipeline saved to unsupervised_model_bundle.pkl")
print("✅ DBSCAN model saved to dbscan_model.pkl (for reference only)")
