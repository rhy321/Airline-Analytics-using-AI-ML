# ===============================================================
# running.py — Unified Supervised + Unsupervised Prediction Script
# ===============================================================

import pandas as pd
import joblib
import pickle

# ---------------------------------------------------------------
# 1. Load Supervised Model (joblib)
# ---------------------------------------------------------------
print("Loading supervised model (joblib)...")
try:
    supervised_model = joblib.load("phase3_best_pipeline.pkl")
    print("✅ Supervised model loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading supervised model: {e}\n")
    supervised_model = None

# ---------------------------------------------------------------
# 2. Load Unsupervised Model (pickle)
# ---------------------------------------------------------------
print("Loading unsupervised model (pickle)...")
try:
    with open("unsupervised_model_bundle.pkl", "rb") as f:
        unsupervised_model = pickle.load(f)
    print("✅ Unsupervised model loaded successfully.\n")
except Exception as e:
    print(f"❌ Error loading unsupervised model: {e}\n")
    unsupervised_model = None

# ---------------------------------------------------------------
# 3. Sample Input Data (example)
# ---------------------------------------------------------------
sample_data = pd.DataFrame({
    "AIRLINE_CODE": ["DL", "AA"],
    "MONTH": [7, 12],
    "DAY_OF_WEEK": [5, 1],
    "DISTANCE": [980, 450],
    "SCHEDULED_DEPARTURE": [830, 2200]
})
print("🧩 Sample input data:")
print(sample_data, "\n")

# ---------------------------------------------------------------
# 4. Supervised Model Predictions (Fixed)
# ---------------------------------------------------------------
supervised_results = sample_data.copy()

if supervised_model is not None:
    try:
        # ✅ Ensure only the features expected by the supervised model are passed
        expected_features = ["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "DISTANCE", "SCHEDULED_DEPARTURE"]
        X_supervised = supervised_results[expected_features]

        preds = supervised_model.predict(X_supervised)

        if hasattr(supervised_model, "predict_proba"):
            probs = supervised_model.predict_proba(X_supervised)[:, 1]
            supervised_results["Delay_Prob"] = probs
        else:
            supervised_results["Delay_Prob"] = None

        supervised_results["Predicted_Delay"] = preds

        print("🔹 Supervised Model Predictions:")
        print(supervised_results[["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "Predicted_Delay", "Delay_Prob"]], "\n")

    except Exception as e:
        print(f"⚠️ Error predicting with supervised model: {e}\n")
        supervised_results["Predicted_Delay"] = None
        supervised_results["Delay_Prob"] = None
else:
    supervised_results["Predicted_Delay"] = None
    supervised_results["Delay_Prob"] = None


# ---------------------------------------------------------------
# 5. Unsupervised Model Clustering
# ---------------------------------------------------------------
if unsupervised_model is not None:
    try:
        # Ensure same input features used during training
        X_unsupervised = sample_data[["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "DISTANCE", "SCHEDULED_DEPARTURE"]]

        # Predict cluster labels using full pipeline
        cluster_labels = unsupervised_model.named_steps["kmeans"].predict(
            unsupervised_model.named_steps["pca"].transform(
                unsupervised_model.named_steps["preprocessor"].transform(X_unsupervised)
            )
        )

        supervised_results["Cluster_Label"] = cluster_labels

        print("🔹 Unsupervised Model Clustering:")
        print(supervised_results[["AIRLINE_CODE", "MONTH", "DAY_OF_WEEK", "Cluster_Label"]], "\n")

    except Exception as e:
        print(f"⚠️ Error predicting with unsupervised model: {e}\n")
        supervised_results["Cluster_Label"] = None
else:
    supervised_results["Cluster_Label"] = None

# ---------------------------------------------------------------
# 6. Final Combined Output
# ---------------------------------------------------------------
print("✅ Final Combined Results:")
print(supervised_results, "\n")

# Optionally, save results
supervised_results.to_csv("final_predictions_output.csv", index=False)
print("📁 Saved combined results to final_predictions_output.csv")
