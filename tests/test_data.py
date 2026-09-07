import pandas as pd
from pathlib import Path
from src.data.generate import generate_synthetic_data

def test_generate_synthetic_data_shape():
    # Vérifie que la fonction génère le bon nombre de lignes et colonnes
    df = generate_synthetic_data(num_samples=100)
    assert df.shape[0] == 100
    assert df.shape[1] == 13

def test_data_columns():
    # Vérifie la présence des colonnes obligatoires, y compris la cible 'default'
    df = generate_synthetic_data(num_samples=50)
    expected_columns = [
        "customer_id", "age", "income", "employment_years", "loan_amount",
        "loan_duration", "credit_score", "number_of_previous_loans",
        "debt_ratio", "number_of_late_payments", "marital_status",
        "education_level", "default"
    ]
    for col in expected_columns:
        assert col in df.columns

def test_no_missing_values():
    # Vérifie qu'il n'y a pas de valeurs nulles (NaN) dans le dataset généré
    df = generate_synthetic_data(num_samples=100)
    assert df.isnull().sum().sum() == 0