import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


def preprocess_data(input_path="data/raw/credit_data.csv", test_size=0.2, random_state=42):
    print("Chargement des données brutes...")
    df = pd.read_csv(input_path)

    # Séparation Features (X) et Cible (y)
    # On retire customer_id et default des features
    X = df.drop(columns=["customer_id", "default"])
    y = df["default"]

    # Identification des types de colonnes
    numeric_features = [
        "age", "income", "employment_years", "loan_amount", 
        "loan_duration", "credit_score", "number_of_previous_loans", 
        "debt_ratio", "number_of_late_payments"
    ]
    categorical_features = ["marital_status", "education_level"]

    # Création du préprocesseur (Pipeline de transformation)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    print("Séparation Train / Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print("Application du prétraitement (Fit sur Train, Transform sur Train/Test)...")
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Récupération des noms de colonnes après OneHotEncoder pour plus de propreté
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_features = numeric_features + encoded_cat_features

    # Transformation en DataFrames pour l'export
    X_train_df = pd.DataFrame(X_train_transformed, columns=all_features)
    X_test_df = pd.DataFrame(X_test_transformed, columns=all_features)

    X_train_df["default"] = y_train.values
    X_test_df["default"] = y_test.values

    # Sauvegarde dans data/processed/
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    preprocessor_path = output_dir / "preprocessor.joblib"

    X_train_df.to_csv(train_path, index=False)
    X_test_df.to_csv(test_path, index=False)
    
    # On sauvegarde aussi le préprocesseur entraîné (nécessaire pour l'API plus tard !)
    joblib.dump(preprocessor, preprocessor_path)

    print(f"Données prétraitées sauvegardées avec succès :")
    print(f" - Train : {train_path} ({X_train_df.shape})")
    print(f" - Test  : {test_path} ({X_test_df.shape})")
    print(f" - Préprocesseur : {preprocessor_path}")


if __name__ == "__main__":
    preprocess_data()