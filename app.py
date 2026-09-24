"""
FraudShield - Flask Backend API
================================
Serves vehicle-insurance fraud predictions from the trained ExtraTrees model
bundle (fraud_pipeline 1.0.0).

MODEL
-----
Trained with an end-to-end leakage-free pipeline on the vehicle insurance claims
dataset (1,000 claims, 24.7% fraudulent).

The saved object is a complete composite pipeline (`ResamplingPipeline`),
performing feature engineering, imputation, and one-hot encoding internally.
The API accepts raw, human-readable claim values as submitted by the frontend.
"""

import json
import os
import sys
import warnings

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(BASE_DIR, "vehicle_insurance_fraud_model_final.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "fraud_model_metrics.json")
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# ?? Load model bundle ????????????????????????????????????????
bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
MODEL_NAME = "ExtraTrees Sampling"  # Override with display name
THRESHOLD = float(bundle.get("threshold", 0.50))
RAW_FEATURE_NAMES = list(bundle.get("feature_names", []))

print(f"[OK] Model loaded: {MODEL_NAME}")
print(f"     Raw features expected: {len(RAW_FEATURE_NAMES)} | decision threshold: {THRESHOLD:.3f}")

# ?? Baseline defaults for unsubmitted fields ?????????????????
DEFAULT_CLAIM = {
    "months_as_customer": 180,
    "age": 38,
    "policy_number": 999999,
    "policy_bind_date": "2014-01-01",
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_deductable": 1000,
    "policy_annual_premium": 1250.0,
    "umbrella_limit": 0,
    "insured_zip": 450000,
    "insured_sex": "MALE",
    "insured_education_level": "College",
    "insured_occupation": "prof-specialty",
    "insured_hobbies": "reading",
    "insured_relationship": "husband",
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_date": "2015-02-01",
    "incident_type": "Multi-vehicle Collision",
    "collision_type": "Front Collision",
    "incident_severity": "Minor Damage",
    "authorities_contacted": "Police",
    "incident_state": "OH",
    "incident_city": "Columbus",
    "incident_location": "Main Street",
    "incident_hour_of_the_day": 12,
    "number_of_vehicles_involved": 2,
    "property_damage": "NO",
    "bodily_injuries": 0,
    "witnesses": 1,
    "police_report_available": "YES",
    "total_claim_amount": 45000,
    "injury_claim": 5000,
    "property_claim": 5000,
    "vehicle_claim": 35000,
    "auto_make": "Toyota",
    "auto_model": "Camry",
    "auto_year": 2010,
    "_c39": None,
}

NUMERIC_FIELDS = [
    "months_as_customer", "age", "policy_deductable", "policy_annual_premium",
    "umbrella_limit", "capital-gains", "capital-loss", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "bodily_injuries", "witnesses",
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    "auto_year",
]

CATEGORY_OPTIONS = {
    "incident_severity": ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"],
    "incident_type": ["Single Vehicle Collision", "Multi-vehicle Collision", "Vehicle Theft", "Parked Car"],
    "collision_type": ["Front Collision", "Rear Collision", "Side Collision", "Unknown"],
    "authorities_contacted": ["Police", "Fire", "Ambulance", "Other", "None"],
    "police_report_available": ["YES", "NO", "Unknown"],
    "property_damage": ["YES", "NO", "Unknown"],
    "incident_state": ["OH", "NY", "SC", "VA", "WV", "NC", "PA"],
    "policy_state": ["OH", "IL", "IN"],
    "policy_csl": ["100/300", "250/500", "500/1000"],
    "insured_sex": ["MALE", "FEMALE"],
    "insured_education_level": ["High School", "College", "Associate", "Masters", "JD", "MD", "PhD"],
    "insured_occupation": [
        "craft-repair", "prof-specialty", "exec-managerial", "sales", "tech-support",
        "protective-serv", "transport-moving", "handlers-cleaners", "machine-op-inspct",
        "adm-clerical", "farming-fishing", "priv-house-serv", "armed-forces", "other-service",
    ],
    "insured_hobbies": [
        "chess", "cross-fit", "golf", "reading", "yachting", "polo", "sleeping", "movies",
        "hiking", "camping", "basketball", "video-games", "dancing", "board-games", "kayaking",
        "paintball", "bungie-jumping", "skydiving", "exercise",
    ],
    "insured_relationship": ["husband", "wife", "own-child", "unmarried", "other-relative", "not-in-family"],
    "auto_make": [
        "Audi", "BMW", "Chevrolet", "Dodge", "Ford", "Honda", "Jeep", "Mercedes", "Nissan",
        "Saab", "Subaru", "Toyota", "Volkswagen", "Accura",
    ],
}


def sanitize_payload(payload: dict) -> pd.DataFrame:
    record = dict(DEFAULT_CLAIM)

    for k, v in payload.items():
        if v is None or v == "":
            continue
        if k in NUMERIC_FIELDS:
            try:
                record[k] = float(v)
            except (ValueError, TypeError):
                pass
        else:
            record[k] = str(v)

    total = float(record.get("total_claim_amount", 0) or 0)
    veh = float(record.get("vehicle_claim", 0) or 0)
    inj = float(record.get("injury_claim", 0) or 0)
    prop = float(record.get("property_claim", 0) or 0)
    if total > 0 and (veh + inj + prop == 0):
        record["vehicle_claim"] = round(total * 0.70)
        record["property_claim"] = round(total * 0.15)
        record["injury_claim"] = round(total * 0.15)

    return pd.DataFrame([record])


# ?? Routes ???????????????????????????????????????????????????
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"message": "No claim data provided."}), 400

        df = sanitize_payload(payload)
        proba_matrix = pipeline.predict_proba(df)
        fraud_prob = float(proba_matrix[0, 1])

        is_fraud = fraud_prob >= THRESHOLD
        if fraud_prob >= THRESHOLD:
            risk = "High"
        elif fraud_prob >= THRESHOLD * 0.6:
            risk = "Medium"
        else:
            risk = "Low"

        return jsonify({
            "prediction": "Fraud" if is_fraud else "Not Fraud",
            "probability": round(fraud_prob, 4),
            "risk_level": risk,
            "threshold": round(THRESHOLD, 3),
            "model": MODEL_NAME,
        }), 200

    except Exception as exc:
        return jsonify({"message": f"Prediction error: {exc}"}), 500


@app.route("/fields", methods=["GET"])
def fields():
    return jsonify({
        "numeric": NUMERIC_FIELDS,
        "categories": CATEGORY_OPTIONS,
        "defaults": DEFAULT_CLAIM,
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    info = {
        "model": MODEL_NAME,
        "threshold": round(THRESHOLD, 3),
        "raw_features": RAW_FEATURE_NAMES,
        "encoded_features": getattr(pipeline, "n_features_in_", len(RAW_FEATURE_NAMES)),
    }
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
            test_m = metrics.get("test_metrics", {})
            info["accuracy"] = test_m.get("Accuracy")
            info["roc_auc"] = test_m.get("ROC_AUC")
            info["pr_auc"] = test_m.get("PR_AUC")
            info["recall"] = test_m.get("Fraud_Recall")
            info["precision"] = test_m.get("Fraud_Precision")
            info["f1"] = test_m.get("Fraud_F1")
            info["confusion_matrix"] = {
                "TN": test_m.get("TN"),
                "FP": test_m.get("FP"),
                "FN": test_m.get("FN"),
                "TP": test_m.get("TP"),
            }
        except Exception:
            pass
    return jsonify(info), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "threshold": round(THRESHOLD, 3),
    }), 200


# ?? Static frontend ???????????????????????????????????????????
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def frontend(path):
    if path and os.path.isfile(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "FraudShield API is running.",
        "endpoints": ["/predict", "/fields", "/model-info", "/health"],
    }), 200


if __name__ == "__main__":
    print(f">> FraudShield API running on http://localhost:5000 ({MODEL_NAME})")
    print("   POST /predict     - submit a claim for fraud scoring")
    print("   GET  /fields      - input schema")
    print("   GET  /model-info  - model + metrics")
    print("   GET  /health      - health check")
    print("   GET  /            - FraudShield web app (frontend/dist)")
    app.run(debug=False, host="0.0.0.0", port=5000)
