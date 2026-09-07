import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

def load_data():
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    X_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    
    X_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]
    
    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs)
    }
    return metrics

def train_and_log_models():
    X_train, y_train, X_test, y_test = load_data()
    
    # Configuration de l'expérience MLflow en local
    mlflow.set_experiment("credit_risk_experiment")
    
    # Dictionnaire des modèles à tester
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Entraînement du modèle : {name}...")
            model.fit(X_train, y_train)
            
            # Évaluation
            metrics = evaluate_model(model, X_test, y_test)
            
            # Logging des paramètres et métriques dans MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metrics(metrics)
            
            # Logging du modèle
            if name == "RandomForest":
                mlflow.sklearn.log_model(model, "model")
            else:
                mlflow.xgboost.log_model(model, "model")
                
            print(f"[{name}] Métriques enregistrées :")
            for metric_name, value in metrics.items():
                print(f" - {metric_name}: {value:.4f}")
                
    # Sauvegarde locale du meilleur modèle par défaut (ex: XGBoost) pour la suite du projet
    best_model = models["XGBoost"]
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "best_model.joblib")
    print("\nMeilleur modèle (XGBoost) sauvegardé localement dans models/best_model.joblib")

if __name__ == "__main__":
    train_and_log_models()