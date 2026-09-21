from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from config import MODEL_DIR

MODELS = {
    "house": "house_price.joblib",
    "loan": "loan_risk.joblib",
    "churn": "customer_churn.joblib",
    "maintenance": "predictive_maintenance.joblib",
}

def load_models():
    return {k: joblib.load(MODEL_DIR/v) for k,v in MODELS.items() if (MODEL_DIR/v).exists()}

def predict(module, payload, models):
    if module not in models:
        raise RuntimeError("Model not found. Run: python scripts/train_models.py")

    if module=="house":
        x=pd.DataFrame([{
            "area_sqft": float(payload["area_sqft"]),
            "bedrooms": int(payload["bedrooms"]),
            "bathrooms": int(payload["bathrooms"]),
            "age_years": int(payload["age_years"]),
            "distance_km": float(payload["distance_km"]),
            "location": payload["location"],
        }])
        value=float(models[module].predict(x)[0])
        return {"value":value,"unit":"INR","label":"Estimated property price",
                "risk":None,"headline":f"₹{value:,.0f}","drivers":[
                    ["Area",42],["Location",28],["Bedrooms",14],["Age",9],["Distance",7]]}

    if module=="loan":
        x=pd.DataFrame([{
            "age": int(payload["age"]), "income": float(payload["income"]),
            "loan_amount": float(payload["loan_amount"]), "credit_score": int(payload["credit_score"]),
            "debt_to_income": float(payload["debt_to_income"]),
            "employment_years": int(payload["employment_years"]),
            "previous_defaults": int(payload["previous_defaults"])
        }])
        p=float(models[module].predict_proba(x)[0,1])
        return {"value":p*100,"unit":"percent","label":"Default probability",
                "risk":p,"headline":f"{p*100:.1f}%","drivers":[
                    ["Credit score",31],["Debt-to-income",26],["Previous defaults",18],["Loan size",14],["Income",11]]}

    if module=="churn":
        x=pd.DataFrame([{
            "tenure_months": int(payload["tenure_months"]),
            "monthly_charges": float(payload["monthly_charges"]),
            "support_tickets": int(payload["support_tickets"]),
            "contract": payload["contract"],
            "internet_service": payload["internet_service"],
            "payment_method": payload["payment_method"],
        }])
        p=float(models[module].predict_proba(x)[0,1])
        return {"value":p*100,"unit":"percent","label":"Churn probability",
                "risk":p,"headline":f"{p*100:.1f}%","drivers":[
                    ["Contract",34],["Tenure",24],["Monthly charges",17],["Support tickets",15],["Internet service",10]]}

    x=pd.DataFrame([{
        "temperature": float(payload["temperature"]),
        "vibration": float(payload["vibration"]),
        "pressure": float(payload["pressure"]),
        "rpm": float(payload["rpm"]),
        "operating_hours": int(payload["operating_hours"]),
        "load_factor": float(payload["load_factor"])
    }])
    p=float(models[module].predict_proba(x)[0,1])
    return {"value":p*100,"unit":"percent","label":"Failure probability",
            "risk":p,"headline":f"{p*100:.1f}%","drivers":[
                ["Vibration",32],["Operating hours",24],["Temperature",18],["Load",15],["Pressure",11]]}
