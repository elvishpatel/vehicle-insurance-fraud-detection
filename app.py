"""
FraudShield — Flask Backend API
Loads the trained Decision Tree model and serves fraud predictions.

IMPORTANT: The model expects 50 features:
- 16 numeric features (already normalized 0-1 in training data)
- 34 one-hot encoded categorical features (Boolean TRUE/FALSE)

The frontend sends raw values (e.g. gender: "Male", age_of_driver: "0.35").
This backend converts them to the model's expected format.
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
CORS(app)  # Enable CORS for React frontend on localhost:3000

# ── Load the trained model ───────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'vehicle_insurance_fraud_detection_model.pkl')
model = joblib.load(MODEL_PATH)
print(f"[OK] Model loaded successfully: {type(model).__name__}")
print(f"   Features expected: {model.n_features_in_}")

# ── The exact feature order the model was trained on ─────────
FEATURE_NAMES = list(model.feature_names_in_)

# Numeric features (frontend sends these as float values 0-1)
NUMERIC_FEATURES = [
    'age_of_driver', 'safety_rating', 'annual_income', 'high_education',
    'address_change', 'past_num_of_claims', 'liab_prct', 'police_report',
    'age_of_vehicle', 'vehicle_price', 'total_claim', 'injury_claim',
    'policy_deductible', 'annual_premium', 'days_open', 'form_defects'
]

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


def transform_input(data):
    """
    Convert raw frontend JSON to a 50-feature numpy array
    matching the model's expected input format.
    """
    # Start with a dict of all features set to 0
    features = {name: 0.0 for name in FEATURE_NAMES}

    # 1) Set numeric features (already expected as 0-1 floats from frontend)
    for field in NUMERIC_FEATURES:
        if field in data:
            try:
                features[field] = float(data[field])
            except (ValueError, TypeError):
                features[field] = 0.0

    # 2) One-hot encode categorical features
    for cat_field, value_map in CATEGORICAL_MAPPINGS.items():
        if cat_field in data:
            selected_value = data[cat_field]
            if selected_value in value_map:
                one_hot_col = value_map[selected_value]
                if one_hot_col in features:
                    features[one_hot_col] = 1.0

    # 3) Build the feature array in the correct column order
    feature_array = np.array([[features[name] for name in FEATURE_NAMES]])
    return feature_array


# ── API Routes ───────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON claim data, transforms it, runs the model,
    and returns prediction + probability.
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'message': 'No input data provided'}), 400

        # Transform the raw input to model-ready features
        features = transform_input(data)

        # Get prediction (0 = Not Fraud, 1 = Fraud)
        prediction = model.predict(features)[0]

        # Get probability scores [P(Not Fraud), P(Fraud)]
        probabilities = model.predict_proba(features)[0]

        # Build response
        result = {
            'prediction': 'Fraud' if prediction == 1 else 'Not Fraud',
            'probability': round(float(probabilities[1]), 2)  # P(Fraud)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': type(model).__name__,
        'features': model.n_features_in_
    }), 200


# ── Run the server ───────────────────────────────────────────
if __name__ == '__main__':
    print("\n>> FraudShield API running on http://localhost:5000")
    print("   POST /predict  — Submit claim for fraud prediction")
    print("   GET  /health   — Health check\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
