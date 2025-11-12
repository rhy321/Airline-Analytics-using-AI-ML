# phase3_training.py
# Run in a Jupyter cell or as a script. Requires scikit-learn, pandas, numpy, matplotlib, seaborn, joblib.

import pandas as pd
import numpy as np
from pathlib import Path

# Modeling / preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# 1) Load data
# -------------------------
data_path = Path("Airline-Analytics-using-AI-ML\data\cleaned_flights.csv")
if not data_path.exists():
    raise FileNotFoundError(f"{data_path} not found - save your cleaned file at this path.")

df = pd.read_csv(data_path)
print(df.columns)
# -------------------------
# 2) Select features + target
# -------------------------
# Edit these as needed for your dataset
FEATURES = ['AIRLINE_CODE', 'MONTH', 'DAY_OF_WEEK', 'DISTANCE', 'SCHEDULED_DEPARTURE']
TARGET = 'DELAYED'   # binary 0/1

# Basic sanity checks
missing_features = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing_features:
    raise ValueError("Missing columns in CSV: " + ", ".join(missing_features))

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

# -------------------------
# 3) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# -------------------------
# 4) Preprocessing pipeline
# -------------------------
# Identify categorical vs numerical features
# Adjust if you have more numeric / categorical columns
numeric_features = [c for c in FEATURES if X_train[c].dtype.kind in "biufc"]
categorical_features = [c for c in FEATURES if c not in numeric_features]

# Numeric transformer: impute (median) + standardize
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical transformer: impute (most_frequent) + one-hot encode
# If high-cardinality (many airlines), consider OrdinalEncoder or target encoding instead
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# -------------------------
# 5) Build model pipelines
# -------------------------
# Logistic Regression baseline
logreg_pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
])

# Random Forest (we will tune some hyperparams)
rf_pipeline = Pipeline(steps=[
    ("preproc", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1))
])

# -------------------------
# 6) Quick cross-validated baseline checks
# -------------------------
print("Cross-validating baselines (ROC AUC, 5-fold)...")
for name, pipe in [("LogisticRegression", logreg_pipeline), ("RandomForest", rf_pipeline)]:
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"{name}: ROC AUC mean={scores.mean():.4f} std={scores.std():.4f}")

# -------------------------
# 7) Fit baseline models
# -------------------------
print("\nFitting baseline Logistic Regression on training set...")
logreg_pipeline.fit(X_train, y_train)

print("Fitting baseline Random Forest on training set...")
rf_pipeline.fit(X_train, y_train)

# -------------------------
# 8) Evaluate on test set
# -------------------------
def evaluate_model(pipeline, X_test, y_test, model_name="Model"):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    print(f"\n=== Evaluation: {model_name} ===")
    print(classification_report(y_test, y_pred, digits=4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    if y_proba is not None:
        roc = roc_auc_score(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        print(f"ROC AUC: {roc:.4f}")
        print(f"Average Precision (PR AUC): {ap:.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc:.3f})")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC Curve — {model_name}")
        plt.legend()
        plt.show()

print("Evaluating Logistic Regression:")
evaluate_model(logreg_pipeline, X_test, y_test, "LogisticRegression")

print("Evaluating Random Forest (baseline):")
evaluate_model(rf_pipeline, X_test, y_test, "RandomForest_Baseline")

# -------------------------
# 9) Hyperparameter tuning for Random Forest (RandomizedSearchCV)
# -------------------------
print("\nStarting RandomizedSearchCV for RandomForest... (will take some time)")

param_dist = {
    "clf__n_estimators": [100, 200, 400],
    "clf__max_depth": [None, 10, 20, 40],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4],
    "clf__max_features": ["sqrt", "log2", 0.5]
}

random_search = RandomizedSearchCV(
    rf_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring="roc_auc",
    n_jobs=-1,
    cv=3,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best RF params:", random_search.best_params_)
print("Best RF CV ROC AUC:", random_search.best_score_)

best_rf_pipeline = random_search.best_estimator_

# Evaluate tuned RF on test set
evaluate_model(best_rf_pipeline, X_test, y_test, "RandomForest_Tuned")

# -------------------------
# 10) Feature importances (for RandomForest)
# -------------------------
# To map back importances to original feature names we need to extract columns output by the preprocessor
def get_feature_names_from_column_transformer(column_transformer):
    output_features = []
    for name, transformer, columns in column_transformer.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "onehot" in transformer.named_steps:
            ohe = transformer.named_steps["onehot"]
            # for scikit-learn >=1.0
            try:
                cols = list(ohe.get_feature_names_out(columns))
            except Exception:
                cols = []
                # fallback: create names manually
                for c in columns:
                    cols.append(c)
            output_features.extend(cols)
        else:
            output_features.extend(columns)
    return output_features

# Only if classifier exposes feature_importances_
clf = best_rf_pipeline.named_steps['clf']
if hasattr(clf, 'feature_importances_'):
    feat_names = get_feature_names_from_column_transformer(best_rf_pipeline.named_steps['preproc'])
    importances = clf.feature_importances_
    if len(feat_names) == len(importances):
        fi = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(30)
        plt.figure(figsize=(8,6))
        sns.barplot(x=fi.values, y=fi.index)
        plt.title("Top feature importances (RandomForest)")
        plt.xlabel("Importance")
        plt.show()
    else:
        print("Could not map importances to feature names (mismatch lengths).")
else:
    print("Best RF model does not expose feature_importances_")

# -------------------------
# 11) Save the best model pipeline
# -------------------------
model_file = "phase3_best_pipeline.pkl"
joblib.dump(best_rf_pipeline, model_file)
print(f"Saved best pipeline to {model_file}")

# -------------------------
# 12) How to load & predict on new data
# -------------------------
# Example:
loaded = joblib.load(model_file)
example_rows = X_test.head(5)
probs = loaded.predict_proba(example_rows)[:, 1]
preds = loaded.predict(example_rows)
print("\nExample predictions on 5 rows:")
print(example_rows.assign(PRED=preds, PROB_DELAY=probs))

# -------------------------
# END
# -------------------------
