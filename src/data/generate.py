import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_data(num_samples=10000, random_state=42):
  np.random.seed(random_state)

  customer_id = [f"CUST_{i:05d}" for i in range(1, num_samples + 1)]
  age = np.random.randint(18, 70, size=num_samples)
  income = np.random.exponential(scale=30000, size=num_samples) + 15000
  income = np.round(income, 2)

  employment_years = np.clip(
      np.random.normal(loc=5, scale=4, size=num_samples), 0, 40
  )
  employment_years = np.round(employment_years, 1)

  loan_amount = np.random.gamma(shape=2, scale=15000, size=num_samples) + 5000
  loan_amount = np.round(loan_amount, 2)

  loan_duration = np.random.choice([12, 24, 36, 48, 60], size=num_samples)

  credit_score = np.random.normal(loc=650, scale=80, size=num_samples)
  credit_score = np.clip(credit_score, 300, 850).astype(int)

  # Utilisation correcte de lam=1.5 pour la loi de Poisson
  number_of_previous_loans = np.random.poisson(lam=1.5, size=num_samples)

  debt_ratio = np.random.beta(a=2, b=5, size=num_samples)
  debt_ratio = np.round(debt_ratio, 4)

  number_of_late_payments = np.random.choice(
      [0, 1, 2, 3, 4, 5], size=num_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]
  )

  marital_status = np.random.choice(
      ["Single", "Married", "Divorced"], size=num_samples, p=[0.4, 0.5, 0.1]
  )
  education_level = np.random.choice(
      ["High School", "Bachelor", "Master", "PhD"],
      size=num_samples,
      p=[0.3, 0.45, 0.2, 0.05],
  )

  default_prob = (
      0.3
      * (debt_ratio > 0.5)
      + 0.3 * (credit_score < 580)
      + 0.2 * (number_of_late_payments > 1)
      + 0.1 * (income < 25000)
      + np.random.normal(0, 0.1, size=num_samples)
  )

  default_prob = 1 / (1 + np.exp(-default_prob * 5))
  default = (default_prob > np.random.uniform(0.3, 0.7, size=num_samples)).astype(
      int
  )

  df = pd.DataFrame({
      "customer_id": customer_id,
      "age": age,
      "income": income,
      "employment_years": employment_years,
      "loan_amount": loan_amount,
      "loan_duration": loan_duration,
      "credit_score": credit_score,
      "number_of_previous_loans": number_of_previous_loans,  # <- Nom de la variable + virgule
      "debt_ratio": debt_ratio,
      "number_of_late_payments": number_of_late_payments,
      "marital_status": marital_status,
      "education_level": education_level,
      "default": default,
  })

  return df


if __name__ == "__main__":
  output_dir = Path("data/raw")
  output_dir.mkdir(parents=True, exist_ok=True)

  print("Génération de la donnée synthétique...")
  df = generate_synthetic_data(num_samples=15000)

  output_path = output_dir / "credit_data.csv"
  df.to_csv(output_path, index=False)
  print(f"Données générées avec succès et sauvegardées dans : {output_path}")
  print(f"Dimensions du dataset : {df.shape}")