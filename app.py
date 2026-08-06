"""
FraudShield — Flask Backend API
Loads the trained Decision Tree model and serves fraud predictions.

AUTOMATIC SCALING ENABLED:
The ML model expects 50 features normalized between 0.0 and 1.0.
The frontend sends natural human inputs (e.g. Age: 35, Income: 60000).
This backend automatically scales human inputs into 0.0 - 1.0 before running prediction!
"""

import os
import warnings
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Suppress version mismatch warning ────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)

# ── Initialize Flask app ─────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# ── Load the trained model ───────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'vehicle_insurance_fraud_detection_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"[OK] Model loaded successfully: {type(model).__name__}")
print(f"   Features expected: {model.n_features_in_}")

# ── The exact feature order the model was trained on ─────────
FEATURE_NAMES = list(model.feature_names_in_)

# Categorical mappings: frontend value → list of one-hot column names
CATEGORICAL_MAPPINGS = {
    'gender': {
        'Female': 'gender_F',
        'Male': 'gender_M',
    },
    'marital_status': {
        'Other': 'marital_status_*',
        'Single': 'marital_status_0',
        'Married': 'marital_status_1',
    },
    'property_status': {
        'Own': 'property_status_Own',
        'Rent': 'property_status_Rent',
    },
    'claim_day_of_week': {
        'Unknown': 'claim_day_of_week_*',
        'Friday': 'claim_day_of_week_Friday',
        'Monday': 'claim_day_of_week_Monday',
        'Saturday': 'claim_day_of_week_Saturday',
        'Sunday': 'claim_day_of_week_Sunday',
        'Thursday': 'claim_day_of_week_Thursday',
        'Tuesday': 'claim_day_of_week_Tuesday',
        'Wednesday': 'claim_day_of_week_Wednesday',
    },
    'accident_site': {
        'Highway': 'accident_site_Highway',
        'Local': 'accident_site_Local',
        'Parking Lot': 'accident_site_Parking Lot',
    },
    'witness_present': {
        'Unknown': 'witness_present_*',
        'No': 'witness_present_0',
        'Yes': 'witness_present_1',
    },
    'channel': {
        'Broker': 'channel_Broker',
        'Online': 'channel_Online',
        'Phone': 'channel_Phone',
    },
    'vehicle_category': {
        'Compact': 'vehicle_category_Compact',
        'Large': 'vehicle_category_Large',
        'Medium': 'vehicle_category_Medium',
    },
    'vehicle_color': {
        'Black': 'vehicle_color_black',
        'Blue': 'vehicle_color_blue',
        'Gray': 'vehicle_color_gray',
        'Other': 'vehicle_color_other',
        'Red': 'vehicle_color_red',
        'Silver': 'vehicle_color_silver',
        'White': 'vehicle_color_white',
    },
}

def scale_numeric_value(field, val):
    """
    Automatically scales natural human input (e.g. Age: 35, Income: $50000)
    into 0.0 - 1.0 range expected by the ML model.
    """
    if val is None or val == '':
        return 0.0

    # Handle Yes / No or string booleans
    if isinstance(val, str):
        val_str = val.strip().lower()
        if val_str in ['yes', 'true', '1']:
            return 1.0
        if val_str in ['no', 'false', '0']:
            return 0.0

    try:
        fval = float(val)
    except (ValueError, TypeError):
        return 0.0

    # Automatic Min-Max Scaling based on realistic dataset bounds
    if field == 'age_of_driver':
        if fval > 1.0:
            return max(0.0, min(1.0, (fval - 18.0) / (74.0 - 18.0)))
        return fval
    elif field == 'safety_rating':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 100.0))
        return fval
    elif field == 'annual_income':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 150000.0))
        return fval
    elif field in ['high_education', 'address_change', 'police_report']:
        return 1.0 if fval >= 0.5 else 0.0
    elif field == 'past_num_of_claims':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 6.0))
        return fval
    elif field == 'liab_prct':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 100.0))
        return fval
    elif field == 'age_of_vehicle':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 10.0))
        return fval
    elif field == 'vehicle_price':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 100000.0))
        return fval
    elif field == 'total_claim':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 100000.0))
        return fval
    elif field == 'injury_claim':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 50000.0))
        return fval
    elif field == 'policy_deductible':
        if fval > 1.0:
            return max(0.0, min(1.0, (fval - 500.0) / (2000.0 - 500.0)))
        return fval
    elif field == 'annual_premium':
        if fval > 1.0:
            return max(0.0, min(1.0, (fval - 500.0) / (3000.0 - 500.0)))
        return fval
    elif field == 'days_open':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 365.0))
        return fval
    elif field == 'form_defects':
        if fval > 1.0:
            return max(0.0, min(1.0, fval / 12.0))
        return fval
    
    return max(0.0, min(1.0, fval))


def transform_input(data):
    """
    Convert raw frontend JSON to a 50-feature numpy array
    matching the model's expected input format.
    """
    features = {name: 0.0 for name in FEATURE_NAMES}

    # 1) Scale numeric & binary features
    for field in FEATURE_NAMES:
        if field in data:
            features[field] = scale_numeric_value(field, data[field])

    # 2) One-hot encode categorical features
    for cat_field, value_map in CATEGORICAL_MAPPINGS.items():
        if cat_field in data:
            selected_value = data[cat_field]
            if selected_value in value_map:
                one_hot_col = value_map[selected_value]
                if one_hot_col in features:
                    features[one_hot_col] = 1.0

    # 3) Build the feature array in exact column order
    feature_array = np.array([[features[name] for name in FEATURE_NAMES]])
    return feature_array


# ── API Routes ───────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({'message': 'No input data provided'}), 400

        # Transform and auto-scale raw input
        features = transform_input(data)

        # Get prediction (0 = Not Fraud, 1 = Fraud)
        prediction = model.predict(features)[0]

        # Get probability scores [P(Not Fraud), P(Fraud)]
        probabilities = model.predict_proba(features)[0]

        result = {
            'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
            'probability': round(float(probabilities[1]), 2)  # P(Fraud)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': type(model).__name__,
        'features': model.n_features_in_
    }), 200


if __name__ == '__main__':
    print("\n>> FraudShield API running on http://localhost:5000")
    print("   POST /predict  — Submit claim for fraud prediction")
    print("   GET  /health   — Health check\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
