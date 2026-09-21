from flask import Flask, render_template, request, jsonify
from pathlib import Path
import sys, subprocess

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from app.predictors import load_models, predict
from app.gemini_service import ask_gemini

app=Flask(__name__)
models=load_models()

MODULES={
 "house":{"title":"Property Price","icon":"⌂","description":"Estimate residential property value from market and property attributes."},
 "loan":{"title":"Loan Risk","icon":"◈","description":"Estimate the probability of a loan default from applicant and loan data."},
 "churn":{"title":"Customer Churn","icon":"◎","description":"Estimate whether a customer is likely to leave."},
 "maintenance":{"title":"Machine Failure","icon":"◌","description":"Estimate equipment failure risk from sensor readings."}
}

@app.get("/")
def index():
    return render_template("index.html", modules=MODULES, ready=len(models)==4)

@app.post("/api/predict/<module>")
def api_predict(module):
    global models
    try:
        if len(models)<4:
            models=load_models()
        payload=request.get_json(force=True)
        result=predict(module,payload,models)
        return jsonify({"ok":True,"result":result})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),400

@app.post("/api/ai")
def api_ai():
    body=request.get_json(force=True)
    result=ask_gemini(body.get("module","unknown"),body.get("payload",{}),body.get("prediction",{}))
    return jsonify(result)

@app.get("/api/health")
def health():
    return jsonify({"ok":True,"models_loaded":len(models),"gemini_configured":bool(__import__("config").GEMINI_API_KEY)})

if __name__=="__main__":
    if len(models)<4:
        print("Models missing. Run: python scripts/train_models.py")
    app.run(host="127.0.0.1",port=5000,debug=True)
