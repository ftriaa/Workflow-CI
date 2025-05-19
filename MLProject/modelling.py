import os
import json
import warnings
import argparse

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="stroke_dataset_preprocessing.csv")
parser.add_argument("--artefak_dir", type=str, default="artefak")
args = parser.parse_args()

# Buat folder artefak jika belum ada
os.makedirs(args.artefak_dir, exist_ok=True)

# Load data
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    return X, y

# Oversampling dengan SMOTE
def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Evaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0,
        "conf_matrix": confusion_matrix(y_test, y_pred)
    }
    return metrics

# Simpan confusion matrix
def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = os.path.join(args.artefak_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path

# Simpan metrik ke JSON
def save_metrics_json(metrics, model_name):
    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        return o

    cleaned_metrics = {k: convert(v) for k, v in metrics.items()}
    path = os.path.join(args.artefak_dir, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(cleaned_metrics, f, indent=4)
    return path

# Training + Tuning + Logging
def tune_and_log_model(name, model, param_grid, X_res, y_res, X_test, y_test):
    with mlflow.start_run(run_name=f"Tuned_{name}", nested=True):
        # Tuning
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_res, y_res)
        best_model = grid.best_estimator_

        # Evaluasi
        scores = evaluate_model(best_model, X_test, y_test)
        cm = scores["conf_matrix"]

        # Simpan confusion matrix & metrik JSON ke folder artefak/
        cm_path = save_confusion_matrix(cm, name)
        json_path = save_metrics_json(scores, name)

        # Logging ke MLflow
        mlflow.log_param("model", name)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics({
            "accuracy": scores["accuracy"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1_score": scores["f1_score"],
            "roc_auc": scores["roc_auc"],
            "conf_matrix_TN": cm[0][0],
            "conf_matrix_FP": cm[0][1],
            "conf_matrix_FN": cm[1][0],
            "conf_matrix_TP": cm[1][1],
        })

        # Log dan simpan model
        mlflow.sklearn.log_model(sk_model=best_model, artifact_path=f"model_{name}")

        print(f"\n{name} - DONE")
        print(f"Best Params: {grid.best_params_}")
        print(f"F1 Score: {scores['f1_score']:.4f} | ROC AUC: {scores['roc_auc']:.4f}")

# Main eksekusi
def main():
    X, y = load_data(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_res, y_res = apply_smote(X_train, y_train)

    models = {
        "RandomForest": {
            "estimator": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 10]}
        },
        "XGBoost": {
            "estimator": XGBClassifier(random_state=42, eval_metric="logloss"),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.01]}
        },
        "LightGBM": {
            "estimator": LGBMClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.1, 0.01]}
        },
        "SVM": {
            "estimator": SVC(probability=True, random_state=42),
            "params": {"C": [1, 10], "kernel": ["rbf", "linear"]}
        },
        "NaiveBayes": {
            "estimator": GaussianNB(),
            "params": {}
        }
    }

    for name, cfg in models.items():
        tune_and_log_model(name, cfg["estimator"], cfg["params"], X_res, y_res, X_test, y_test)

if __name__ == "__main__":
    main()
