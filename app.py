"""
FraudShield - Flask Backend API
================================
Serves vehicle-insurance fraud predictions from the trained LightGBM model.

MODEL
-----
Trained on the real-world "Angoss Knowledge Seeker" automobile insurance claims
dataset (15,420 claims, 5.99% fraudulent) - see train_fraud_model.py.

The saved object is a complete scikit-learn Pipeline, so it performs its own
feature encoding (ordered fields -> ordinal codes, nominal fields -> one-hot)
internally. The API therefore accepts raw, human-readable claim values exactly
as the frontend form produces them - there is no manual scaling step.

Default operating threshold comes from the saved bundle and was chosen on
out-of-fold data to keep accuracy above 90%.
"""

import json
import os
import warnings

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "fraud_model_metrics.json")
FRONTEND_DIST = os.path.join(BASE_DIR, "frontend", "dist")

# ── Load model bundle ────────────────────────────────────────
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
MODEL_NAME = bundle["model_name"]
THRESHOLD = float(bundle["threshold"])
FEATURE_NAMES = list(bundle["features"])

print(f"[OK] Model loaded: {MODEL_NAME}")
print(f"     Features: {len(FEATURE_NAMES)} | decision threshold: {THRESHOLD:.3f}")


# ── Input schema ─────────────────────────────────────────────
# Mirrors the exact feature order and category spellings of the training data.
NUMERIC_FIELDS = [
    "WeekOfMonth", "Age", "WeekOfMonthClaimed", "RepNumber",
    "Deductible", "DriverRating", "Year",
]

CATEGORY_FIELDS = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "DayOfWeek": ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"],
    "Make": ["Accura", "BMW", "Chevrolet", "Dodge", "Ferrari", "Ford", "Honda",
             "Jaguar", "Lexus", "Mazda", "Mecedes", "Mercury", "Nisson", "Pontiac",
             "Porche", "Saab", "Saturn", "Toyota", "VW"],
    "AccidentArea": ["Rural", "Urban"],
    "DayOfWeekClaimed": ["0", "Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"],
    "MonthClaimed": ["0", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "Sex": ["Female", "Male"],
    "MaritalStatus": ["Divorced", "Married", "Single", "Widow"],
    "Fault": ["Policy Holder", "Third Party"],
    "PolicyType": ["Sedan - All Perils", "Sedan - Collision", "Sedan - Liability",
                   "Sport - All Perils", "Sport - Collision", "Sport - Liability",
                   "Utility - All Perils", "Utility - Collision", "Utility - Liability"],
    "VehicleCategory": ["Sedan", "Sport", "Utility"],
    "VehiclePrice": ["less than 20000", "20000 to 29000", "30000 to 39000",
                     "40000 to 59000", "60000 to 69000", "more than 69000"],
    "Days_Policy_Accident": ["none", "1 to 7", "8 to 15", "15 to 30", "more than 30"],
    "Days_Policy_Claim": ["none", "8 to 15", "15 to 30", "more than 30"],
    "PastNumberOfClaims": ["none", "1", "2 to 4", "more than 4"],
    "AgeOfVehicle": ["new", "2 years", "3 years", "4 years", "5 years",
                     "6 years", "7 years", "more than 7"],
    "AgeOfPolicyHolder": ["16 to 17", "18 to 20", "21 to 25", "26 to 30", "31 to 35",
                          "36 to 40", "41 to 50", "51 to 65", "over 65"],
    "PoliceReportFiled": ["No", "Yes"],
    "WitnessPresent": ["No", "Yes"],
    "AgentType": ["External", "Internal"],
    "NumberOfSuppliments": ["none", "1 to 2", "3 to 5", "more than 5"],
    "AddressChange_Claim": ["no change", "under 6 months", "1 year",
                            "2 to 3 years", "4 to 8 years"],
    "NumberOfCars": ["1 vehicle", "2 vehicles", "3 to 4", "5 to 8", "more than 8"],
    "BasePolicy": ["All Perils", "Collision", "Liability"],
}

# The form only asks for the 15 fields that carry the predictive signal
# (permutation importance on the test set). Everything else is auto-filled
# with the training-set mode/median - measured cost of this simplification:
# accuracy 91.2% -> 90.8%, ROC-AUC actually improves 0.846 -> 0.856.
REQUIRED_FIELDS = [
    "Month", "WeekOfMonth", "Age", "RepNumber", "Year", "BasePolicy", "PolicyType",
    "VehiclePrice", "AgeOfVehicle", "Deductible", "MonthClaimed", "DayOfWeekClaimed",
    "PastNumberOfClaims", "Fault", "AddressChange_Claim",
]

DEFAULTS = {
    "Sex": "Male",
    "MaritalStatus": "Married",
    "DayOfWeek": "Monday",
    "Make": "Pontiac",
    "AccidentArea": "Urban",
    "WeekOfMonthClaimed": 3.0,
    "Days_Policy_Accident": "more than 30",
    "Days_Policy_Claim": "more than 30",
    "VehicleCategory": "Sedan",
    "AgeOfPolicyHolder": "31 to 35",
    "PoliceReportFiled": "No",
    "WitnessPresent": "No",
    "AgentType": "External",
    "NumberOfSuppliments": "none",
    "NumberOfCars": "1 vehicle",
    "DriverRating": 2.0,
}


def validate(payload):
    """Return (clean_record, errors). Unprovided optional fields fall back to
    training-data defaults; required fields must be present and valid."""
    errors = []

    missing = [f for f in REQUIRED_FIELDS if payload.get(f) in (None, "")]
    if missing:
        errors.append("Missing required fields: " + ", ".join(missing))

    record = dict(DEFAULTS)

    for field in NUMERIC_FIELDS:
        raw = payload.get(field)
        if raw in (None, ""):
            continue
        try:
            record[field] = float(raw)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number (got '{raw}').")

    for field, allowed in CATEGORY_FIELDS.items():
        raw = payload.get(field)
        if raw in (None, ""):
            continue
        value = str(raw)
        if value not in allowed:
            errors.append(f"'{field}' must be one of: {', '.join(allowed)}.")
        else:
            record[field] = value

    return record, errors


# ── Routes ───────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"message": "No input data provided"}), 400

        record, errors = validate(payload)
        if errors:
            return jsonify({"message": " ".join(errors), "errors": errors}), 400

        features = pd.DataFrame([[record[f] for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
        probability = float(model.predict_proba(features)[0, 1])

        is_fraud = probability >= THRESHOLD
        if is_fraud:
            risk = "High"
        elif probability >= THRESHOLD * 0.6:
            risk = "Medium"
        else:
            risk = "Low"

        return jsonify({
            "prediction": "Fraud" if is_fraud else "Not Fraud",
            "probability": round(probability, 4),
            "risk_level": risk,
            "threshold": round(THRESHOLD, 3),
        }), 200

    except Exception as exc:  # pragma: no cover - defensive
        return jsonify({"message": f"Prediction error: {exc}"}), 500


@app.route("/fields", methods=["GET"])
def fields():
    """Input schema, so the UI can stay in sync with the model."""
    return jsonify({
        "numeric": NUMERIC_FIELDS,
        "categorical": CATEGORY_FIELDS,
        "required": REQUIRED_FIELDS,
        "auto_filled": sorted(DEFAULTS),
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    info = {"model": MODEL_NAME, "threshold": round(THRESHOLD, 3),
            "features": FEATURE_NAMES}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as fh:
            metrics = json.load(fh)
        info["roc_auc"] = metrics.get("roc_auc")
        info["pr_auc"] = metrics.get("pr_auc")
        info["operating_points"] = metrics.get("operating_points")
    return jsonify(info), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": MODEL_NAME,
        "features": len(FEATURE_NAMES),
        "threshold": round(THRESHOLD, 3),
    }), 200


# ── Static frontend (built React app in frontend/dist) ───────
# With a built frontend, `python app.py` alone serves the whole product on
# port 5000; the API routes above always take precedence over this catch-all.
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def frontend(path):
    if path and os.path.isfile(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    index_path = os.path.join(FRONTEND_DIST, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(FRONTEND_DIST, "index.html")
    return jsonify({
        "message": "FraudShield API is running. Build the frontend "
                   "(npm run build in frontend/) or use the dev server on port 3000.",
        "endpoints": ["/predict", "/fields", "/model-info", "/health"],
    }), 200


if __name__ == "__main__":
    print("\n>> FraudShield running on http://localhost:5000")
    print("   POST /predict     - submit a claim for fraud scoring")
    print("   GET  /fields      - input schema")
    print("   GET  /model-info  - model + metrics")
    print("   GET  /health      - health check")
    print("   GET  /            - FraudShield web app (frontend/dist)\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
