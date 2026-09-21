from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import MODEL_DIR, DATA_DIR

rng = np.random.default_rng(42)

def build_preprocessor(cat, num):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num)
    ])

def train_house(n=1600):
    area = rng.integers(450, 4200, n)
    beds = rng.integers(1, 6, n)
    baths = np.clip(beds + rng.integers(-1, 2, n), 1, 6)
    age = rng.integers(0, 31, n)
    distance = rng.uniform(0.5, 25, n)
    location = rng.choice(["Prime", "Urban", "Suburban", "Outer"], n, p=[.18,.35,.32,.15])
    loc_bonus = {"Prime": 2600000, "Urban": 1200000, "Suburban": 450000, "Outer": 0}
    price = (area*7200 + beds*850000 + baths*450000 - age*95000
             - distance*45000 + np.array([loc_bonus[x] for x in location])
             + rng.normal(0, 550000, n))
    price = np.maximum(price, 600000)
    df = pd.DataFrame({
        "area_sqft": area, "bedrooms": beds, "bathrooms": baths, "age_years": age,
        "distance_km": distance.round(2), "location": location, "price": price.round(0)
    })
    df.to_csv(DATA_DIR/"house_prices.csv", index=False)
    X=df.drop(columns="price"); y=df.price
    cat=["location"]; num=[c for c in X.columns if c not in cat]
    pipe=Pipeline([("prep",build_preprocessor(cat,num)),
                   ("model",RandomForestRegressor(n_estimators=220,random_state=42,n_jobs=-1,min_samples_leaf=2))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    pipe.fit(Xtr,ytr)
    pred=pipe.predict(Xte)
    joblib.dump(pipe, MODEL_DIR/"house_price.joblib")
    return {"module":"house","metric":f"R² {r2_score(yte,pred):.3f}","rows":n}

def train_loan(n=2200):
    age=rng.integers(20,70,n)
    income=rng.integers(18000,240000,n)
    loan=rng.integers(10000,1500000,n)
    credit=rng.integers(420,850,n)
    dti=rng.uniform(.05,.75,n)
    emp=rng.integers(0,25,n)
    prior=rng.integers(0,4,n)
    score=(-0.012*(credit-600)+0.000006*loan-0.000004*income+2.5*dti
           +0.75*prior-0.045*emp+rng.normal(0,.9,n))
    prob=1/(1+np.exp(-score))
    default=rng.binomial(1, np.clip(prob,.02,.9))
    df=pd.DataFrame({"age":age,"income":income,"loan_amount":loan,"credit_score":credit,
                     "debt_to_income":dti.round(3),"employment_years":emp,
                     "previous_defaults":prior,"default":default})
    df.to_csv(DATA_DIR/"loan_risk.csv",index=False)
    X=df.drop(columns="default"); y=df.default
    pipe=Pipeline([("scale",StandardScaler()),
                   ("model",GradientBoostingClassifier(random_state=42,n_estimators=160,max_depth=3))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte); proba=pipe.predict_proba(Xte)[:,1]
    joblib.dump(pipe, MODEL_DIR/"loan_risk.joblib")
    return {"module":"loan","metric":f"ROC-AUC {roc_auc_score(yte,proba):.3f}","rows":n}

def train_churn(n=2400):
    tenure=rng.integers(1,73,n)
    monthly=rng.uniform(20,180,n)
    support=rng.integers(0,10,n)
    contract=rng.choice(["Month-to-month","One year","Two year"],n,p=[.55,.25,.20])
    internet=rng.choice(["Fiber","DSL","None"],n,p=[.55,.35,.10])
    payment=rng.choice(["Electronic","Card","Bank transfer"],n)
    risk=1.1 - .035*tenure + .012*monthly + .18*support
    risk += np.where(contract=="Month-to-month",.9,np.where(contract=="One year",.1,-.55))
    risk += np.where(internet=="Fiber",.2,np.where(internet=="DSL",0,-.25))
    prob=1/(1+np.exp(-(risk-1.2)))
    churn=rng.binomial(1,np.clip(prob,.03,.95))
    df=pd.DataFrame({"tenure_months":tenure,"monthly_charges":monthly.round(2),
                     "support_tickets":support,"contract":contract,"internet_service":internet,
                     "payment_method":payment,"churn":churn})
    df.to_csv(DATA_DIR/"customer_churn.csv",index=False)
    X=df.drop(columns="churn"); y=df.churn
    cat=["contract","internet_service","payment_method"]
    num=[c for c in X.columns if c not in cat]
    pipe=Pipeline([("prep",build_preprocessor(cat,num)),
                   ("model",RandomForestClassifier(n_estimators=240,random_state=42,class_weight="balanced",n_jobs=-1,min_samples_leaf=2))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    joblib.dump(pipe,MODEL_DIR/"customer_churn.joblib")
    return {"module":"churn","metric":f"Accuracy {accuracy_score(yte,pred):.3f}","rows":n}

def train_maintenance(n=2500):
    temp=rng.normal(68,12,n)
    vibration=np.abs(rng.normal(.42,.18,n))
    pressure=rng.normal(32,5,n)
    rpm=rng.normal(1800,250,n)
    hours=rng.integers(50,12000,n)
    load=rng.uniform(.25,1,n)
    risk=(.035*(temp-65)+2.3*(vibration-.35)+.025*(pressure-32)
          +.00012*(hours-3000)+.7*(load-.7)+rng.normal(0,.45,n))
    prob=1/(1+np.exp(-risk))
    failure=rng.binomial(1,np.clip(prob,.02,.9))
    df=pd.DataFrame({"temperature":temp.round(2),"vibration":vibration.round(3),
                     "pressure":pressure.round(2),"rpm":rpm.round(0),"operating_hours":hours,
                     "load_factor":load.round(3),"failure":failure})
    df.to_csv(DATA_DIR/"maintenance.csv",index=False)
    X=df.drop(columns="failure"); y=df.failure
    pipe=Pipeline([("scale",StandardScaler()),
                   ("model",RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced",n_jobs=-1,min_samples_leaf=2))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
    joblib.dump(pipe,MODEL_DIR/"predictive_maintenance.joblib")
    return {"module":"maintenance","metric":f"Accuracy {accuracy_score(yte,pred):.3f}","rows":n}

if __name__=="__main__":
    results=[train_house(),train_loan(),train_churn(),train_maintenance()]
    print("\nPredictAI model training complete:")
    for r in results: print(r)
