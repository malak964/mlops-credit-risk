from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"

def test_predict_endpoint():
    # Données valides d'un demandeur de prêt
    payload = {
        "age": 35,
        "income": 50000.0,
        "employment_years": 5.0,
        "loan_amount": 15000.0,
        "loan_duration": 36,
        "credit_score": 700,
        "number_of_previous_loans": 1,
        "debt_ratio": 0.25,
        "number_of_late_payments": 0,
        "marital_status": "Married",
        "education_level": "Bachelor"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "default_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["High Risk", "Low Risk"]