# ===============================================================
# Run Both Models (Supervised + Unsupervised)
# ===============================================================

import pandas as pd
import joblib
import pickle

# ---------------------------------------------------------------
# 1️⃣ Load Supervised Model (Joblib)
# ---------------------------------------------------------------
supervised_model_path = "phase3_best_pipeline.pkl"
print("Loading supervised model (joblib)...")

try:
    sup_model = joblib.load(supervised_model_path)
    print("✅ Supervised model loaded successfully.\n")
except Exception as e:
    print("❌ Error loading supervised model:", e)
    sup_model = None

# ---------------------------------------------------------------
# 2️⃣ Load Unsupervised Model (Pickle)
# ---------------------------------------------------------------

unsupervised_model_path = "unsupervised_model_bundle.pkl"   # your saved pickle
print("Loading unsupervised model (pickle)...")

try:
    with open(unsupervised_model_path, "rb") as f:
        unsup_bundle = pickle.load(f)
    print("✅ Unsupervised data loaded successfully.\n")

    # Extract models if present in the dict
    unsup_model = unsup_bundle.get("kmeans")
    scaler = unsup_bundle.get("scaler")
    pca = unsup_bundle.get("pca")

    if unsup_model is None:
        raise ValueError("KMeans model not found inside pickle.")
except Exception as e:
    print("❌ Error loading unsupervised model:", e)
    unsup_model = None


# ---------------------------------------------------------------
# 3️⃣ Create Sample Input for Prediction
# ---------------------------------------------------------------
sample_input = pd.DataFrame([
    {"AIRLINE_CODE": "DL", "MONTH": 7, "DAY_OF_WEEK": 5, "DISTANCE": 980, "SCHEDULED_DEPARTURE": 830},
    {"AIRLINE_CODE": "AA", "MONTH": 12, "DAY_OF_WEEK": 1, "DISTANCE": 450, "SCHEDULED_DEPARTURE": 2200}
])

print("🧩 Sample input data:")
print(sample_input)

# ---------------------------------------------------------------
# 4️⃣ Predict using Supervised Model
# ---------------------------------------------------------------

if sup_model:
    print("\n🔹 Supervised Model Predictions:")
    try:
        preds = sup_model.predict(sample_input)
        if hasattr(sup_model, "predict_proba"):
            probs = sup_model.predict_proba(sample_input)[:, 1]
            sample_input["Delay_Prob"] = probs
        sample_input["Predicted_Delay"] = preds
        print(sample_input[["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "Predicted_Delay", "Delay_Prob"]])
    except Exception as e:
        print("⚠️ Error predicting with supervised model:", e)
else:
    print("⚠️ Supervised model not loaded, skipping prediction.")

# ---------------------------------------------------------------
# 5️⃣ Predict/Cluster using Unsupervised Model
# ---------------------------------------------------------------

if unsup_model:
    print("\n🔹 Unsupervised Model Clustering:")
    try:
        # Extract numeric features
        numeric_data = sample_input.select_dtypes(include="number")

        # Use same preprocessing as training (you can reapply scaler + PCA here)
        cluster_labels = unsup_model.predict(numeric_data)
        sample_input["Cluster_Label"] = cluster_labels
        print(sample_input[["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "Cluster_Label"]])
    except Exception as e:
        print("⚠️ Error predicting with unsupervised model:", e)
else:
    print("⚠️ Unsupervised model not loaded, skipping clustering.")


# ---------------------------------------------------------------
# 6️⃣ Final Output
# ---------------------------------------------------------------
print("\n✅ Final Combined Results:")
print(sample_input)
