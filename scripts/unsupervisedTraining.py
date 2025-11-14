# ===============================================================
# Phase 3: Unsupervised Model Training – Simplified Airline Clustering
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import pickle

# ---------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------
df = pd.read_csv("data/cleaned_flights.csv")  # Replace with your file path
print("Original Shape:", df.shape)

# ---------------------------------------------------------------
# 2. Select relevant features for clustering
# ---------------------------------------------------------------
features = ["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "DISTANCE", "SCHEDULED_DEPARTURE"]
df_small = df[features].dropna()
print("Selected columns:", df_small.columns.tolist())

# ---------------------------------------------------------------
# 3. Define preprocessing pipeline
# ---------------------------------------------------------------
numeric_features = ["MONTH", "DAY_OF_WEEK", "DISTANCE", "SCHEDULED_DEPARTURE"]
categorical_features = ["AIRLINE_CODE"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ---------------------------------------------------------------
# 4. Build pipeline with PCA and KMeans
# ---------------------------------------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
pca = PCA(n_components=2, random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("pca", pca),
    ("kmeans", kmeans)
])

# ---------------------------------------------------------------
# 5. Fit the pipeline
# ---------------------------------------------------------------
X = df_small
pipeline.fit(X)

# Get transformed data and cluster labels
X_pca = pipeline.named_steps["pca"].transform(
    pipeline.named_steps["preprocessor"].transform(X)
)
labels = pipeline.named_steps["kmeans"].labels_

sil_score = silhouette_score(X_pca, labels)
print(f"\n✅ K-Means Silhouette Score: {sil_score:.4f}")

# ---------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------
df_result = df_small.copy()
df_result["Cluster_Label"] = labels
df_result.to_csv("unsupervised_clusters_output.csv", index=False)
print("✅ Clustered data saved to: unsupervised_clusters_output.csv")

# ---------------------------------------------------------------
# 7. Save model pipeline
# ---------------------------------------------------------------
with open("unsupervised_model_bundle.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Full unsupervised pipeline saved to: unsupervised_model_bundle.pkl")

# ---------------------------------------------------------------
# 8. (Optional) DBSCAN for analysis
# ---------------------------------------------------------------
dbscan = DBSCAN(eps=0.5, min_samples=10)
db_labels = dbscan.fit_predict(X_pca)

unique_db = set(db_labels)
if len(unique_db) > 1 and -1 not in unique_db:
    sil_db = silhouette_score(X_pca, db_labels)
else:
    sil_db = "N/A"

print(f"✅ DBSCAN Silhouette Score: {sil_db}")
print("DBSCAN clusters found:", unique_db)
