import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load models
sup_model = joblib.load("phase3_best_pipeline.pkl")

with open("unsupervised_model_bundle.pkl", "rb") as f:
    unsup_bundle = pickle.load(f)

kmeans = unsup_bundle["kmeans"]
pca = unsup_bundle["pca"]

# Load clustered data
df_clusters = pd.read_csv("unsupervised_clusters_output.csv")

# Detect which columns PCA expects
columns_used = unsup_bundle.get("columns_used", None)

if columns_used:
    numeric_df = df_clusters[columns_used].fillna(0)
else:
    print("⚠️ PCA expected", pca.n_features_in_, "features.")
    print("Selecting the first", pca.n_features_in_, "numeric columns as fallback.")
    numeric_df = df_clusters.select_dtypes(include=np.number).drop(columns=["KMeans_Cluster"], errors="ignore").iloc[:, :pca.n_features_in_].fillna(0)

# Apply PCA
pca_features = pca.transform(numeric_df)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=pca_features[:, 0],
    y=pca_features[:, 1],
    hue=df_clusters["KMeans_Cluster"],
    palette="tab10",
    s=50
)
plt.title("KMeans Clusters (PCA Reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()
