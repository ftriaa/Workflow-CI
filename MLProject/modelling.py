import os
import json
import warnings

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

# Setup MLflow Tracking (Opsional bisa diset di MLproject / ENV) 
mlflow.set_experiment("Model_Tuning_Advanced")

# Folder artefak lokal
ARTEFAK_DIR = "artefak"
os.makedirs(ARTEFAK_DIR, exist_ok=True)

# Load dan Preprocessing Data
def load_data(path="stroke_dataset_preprocessing.csv"):
    df = pd.read_csv(path)
    X = df.drop("stroke", axis=1)
    y = df["stroke"]
    return X, y

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

# Evaluasi & Logging Artefak
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

def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    path = os.path.join(ARTEFAK_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(path)
    plt.close()
    return path

def save_metrics_json(metrics, model_name):
    path = os.path.join(ARTEFAK_DIR, f"{model_name}_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    return path

# Tuning, Training, Logging
def tune_and_log_model(name, model, param_grid, X_res, y_res, X_test, y_test):
    with mlflow.start_run(run_name=f"Tuned_{name}"):
        grid = GridSearchCV(model, param_grid, cv=3, scoring="f1", n_jobs=-1)
        grid.fit(X_res, y_res)
        best_model = grid.best_estimator_

        scores = evaluate_model(best_model, X_test, y_test)
        cm = scores["conf_matrix"]

        # Logging param dan metrik
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

        # Logging model
        mlflow.sklearn.log_model(best_model, "model")

        # Logging artefak visual
        cm_path = save_confusion_matrix(cm, name)
        json_path = save_metrics_json(scores, name)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(json_path)

        print(f"\n{name} - DONE")
        print(f"Best Params: {grid.best_params_}")
        print(f"F1 Score: {scores['f1_score']:.4f} | ROC AUC: {scores['roc_auc']:.4f}")

# Main Eksekusi 
def main():
    X, y = load_data()
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
