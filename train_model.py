"""
================================================================================
Heart Disease Prediction - Model Training & Evaluation Pipeline
================================================================================
This script:
1. Loads and preprocesses the UCI Cleveland Heart Disease dataset.
2. Performs stratified train-test splitting (80/20).
3. Fits StandardScaler on training data and transforms both train and test sets.
4. Trains 4 machine learning classifiers:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - AdaBoost
5. Evaluates models on Accuracy, Precision, Recall, F1-Score, ROC-AUC, and 5-Fold CV.
6. Identifies the best-performing model.
7. Serializes the champion model, scaler, and evaluation metrics using joblib and JSON.
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def load_data(filepath="data/heart.csv"):
    """Load heart disease dataset from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Successfully loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_and_split(df, target_col="target", test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets and scale features.
    Strict featurization ordering: scaler is fitted ONLY on training data.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def get_models(random_state=42):
    """Initialize dictionary of machine learning classification models."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_split=4,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss"
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state
        )
    }
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """
    Train and evaluate all models across multiple metrics.
    Returns metrics dataframe, raw metrics dict, fitted models, and confusion matrices.
    """
    results = []
    fitted_models = {}
    confusion_matrices = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "="*80)
    print(f"{'MODEL EVALUATION & BENCHMARKING':^80}")
    print("="*80)

    for name, model in models.items():
        # Fit model on training data
        model.fit(X_train, y_train)
        fitted_models[name] = model

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)

        # Metrics calculation
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

        cm = confusion_matrix(y_test, y_pred).tolist()
        confusion_matrices[name] = cm

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc_auc, 4),
            "CV_Accuracy_Mean": round(cv_mean, 4),
            "CV_Accuracy_Std": round(cv_std, 4)
        })

        print(f"[{name}]")
        print(f"  Accuracy : {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"  5-Fold CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print("-" * 80)

    metrics_df = pd.DataFrame(results)
    return metrics_df, fitted_models, confusion_matrices

def extract_feature_importances(fitted_models, feature_names):
    """Extract feature importance or coefficients across models."""
    importances = {}
    for name, model in fitted_models.items():
        if hasattr(model, "feature_importances_"):
            importances[name] = dict(zip(feature_names, [round(float(v), 4) for v in model.feature_importances_]))
        elif hasattr(model, "coef_"):
            importances[name] = dict(zip(feature_names, [round(float(abs(v)), 4) for v in model.coef_[0]]))
    return importances

def save_artifacts(best_model_name, fitted_models, scaler, feature_names, metrics_df, confusion_matrices, feature_importances, output_dir="models"):
    """Save trained best model, scaler, and metric artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    best_model = fitted_models[best_model_name]
    
    # Model bundle for Streamlit app
    model_bundle = {
        "best_model_name": best_model_name,
        "model": best_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "all_models": fitted_models,
        "metrics": metrics_df.to_dict(orient="records"),
        "confusion_matrices": confusion_matrices,
        "feature_importances": feature_importances
    }

    pkl_path = os.path.join(output_dir, "heart_disease_model.pkl")
    joblib.dump(model_bundle, pkl_path)
    print(f"\n[INFO] Saved serialized model bundle to: {pkl_path}")

    # Metrics JSON
    json_path = os.path.join(output_dir, "model_metrics.json")
    with open(json_path, "w") as f:
        json.dump({
            "champion_model": best_model_name,
            "metrics": metrics_df.to_dict(orient="records"),
            "confusion_matrices": confusion_matrices,
            "feature_importances": feature_importances
        }, f, indent=4)
    print(f"[INFO] Saved metrics JSON to: {json_path}")

def run_pipeline():
    """Execute complete training and serialization pipeline."""
    df = load_data("data/heart.csv")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_and_split(df)
    
    models = get_models()
    metrics_df, fitted_models, confusion_matrices = evaluate_models(
        models, X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Select champion model based on composite score (F1 + Accuracy + ROC-AUC)
    metrics_df["Composite_Score"] = metrics_df["F1-Score"] + metrics_df["Accuracy"] + metrics_df["ROC-AUC"]
    best_row = metrics_df.sort_values(by="Composite_Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]

    print("\n" + "="*80)
    print(f"CHAMPION MODEL SELECTED: {best_model_name}")
    print(f"Accuracy: {best_row['Accuracy']} | F1-Score: {best_row['F1-Score']} | ROC-AUC: {best_row['ROC-AUC']}")
    print("="*80)

    feature_importances = extract_feature_importances(fitted_models, feature_names)

    save_artifacts(
        best_model_name=best_model_name,
        fitted_models=fitted_models,
        scaler=scaler,
        feature_names=feature_names,
        metrics_df=metrics_df,
        confusion_matrices=confusion_matrices,
        feature_importances=feature_importances,
        output_dir="models"
    )

    print("\n[SUCCESS] Heart Disease ML Training Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
