import React, { useState } from 'react';
import { FaSpinner, FaUserTie, FaFileInvoiceDollar, FaCarCrash, FaMagic } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { predictFraud } from '../services/api';
import './PredictionForm.css';

const SEVERITIES = ['Major Damage', 'Minor Damage', 'Total Loss', 'Trivial Damage'];
const INCIDENT_TYPES = ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Vehicle Theft', 'Parked Car'];
const COLLISION_TYPES = ['Front Collision', 'Rear Collision', 'Side Collision', 'Unknown'];
const AUTHORITIES = ['Police', 'Fire', 'Ambulance', 'Other', 'None'];
const MAKES = [
  'Toyota', 'Ford', 'Chevrolet', 'Dodge', 'Honda', 'BMW', 'Mercedes', 'Audi',
  'Jeep', 'Nissan', 'Saab', 'Subaru', 'Volkswagen', 'Accura'
];

const incidentFields = [
  { name: 'incident_severity', label: 'Incident Severity *', type: 'select', options: ['', ...SEVERITIES] },
  { name: 'incident_type', label: 'Incident Type *', type: 'select', options: ['', ...INCIDENT_TYPES] },
  { name: 'collision_type', label: 'Collision Type', type: 'select', options: ['', ...COLLISION_TYPES] },
  { name: 'number_of_vehicles_involved', label: 'Vehicles Involved', placeholder: '1 - 4', type: 'number' },
  { name: 'incident_hour_of_the_day', label: 'Incident Hour (0 - 23)', placeholder: 'e.g. 14', type: 'number' },
  { name: 'authorities_contacted', label: 'Authorities Contacted', type: 'select', options: ['', ...AUTHORITIES] },
  { name: 'police_report_available', label: 'Police Report Available', type: 'select', options: ['', 'YES', 'NO', 'Unknown'] },
  { name: 'property_damage', label: 'Property Damage Reported', type: 'select', options: ['', 'YES', 'NO', 'Unknown'] },
  { name: 'bodily_injuries', label: 'Bodily Injuries Count', placeholder: '0 - 2', type: 'number' },
  { name: 'witnesses', label: 'Witnesses Count', placeholder: '0 - 3', type: 'number' },
];

const claimFinancialFields = [
  { name: 'total_claim_amount', label: 'Total Claim Amount ($) *', placeholder: 'e.g. 45000', type: 'number' },
  { name: 'vehicle_claim', label: 'Vehicle Claim Amount ($)', placeholder: 'e.g. 35000', type: 'number' },
  { name: 'property_claim', label: 'Property Claim Amount ($)', placeholder: 'e.g. 5000', type: 'number' },
  { name: 'injury_claim', label: 'Injury Claim Amount ($)', placeholder: 'e.g. 5000', type: 'number' },
  { name: 'policy_deductable', label: 'Policy Deductible ($)', type: 'select', options: ['', '500', '1000', '2000'] },
  { name: 'policy_annual_premium', label: 'Annual Premium ($)', placeholder: 'e.g. 1250', type: 'number' },
];

const policyholderFields = [
  { name: 'insured_sex', label: 'Gender', type: 'select', options: ['', 'MALE', 'FEMALE'] },
  { name: 'age', label: 'Policyholder Age', placeholder: 'e.g. 38', type: 'number' },
  { name: 'insured_relationship', label: 'Relationship Status', type: 'select', options: ['', 'husband', 'wife', 'own-child', 'unmarried', 'other-relative', 'not-in-family'] },
  { name: 'auto_make', label: 'Vehicle Make', type: 'select', options: ['', ...MAKES] },
  { name: 'auto_model', label: 'Vehicle Model', placeholder: 'e.g. Camry', type: 'text' },
  { name: 'auto_year', label: 'Vehicle Model Year', placeholder: 'e.g. 2012', type: 'number' },
];

const allFormFields = [...incidentFields, ...claimFinancialFields, ...policyholderFields];

const SAMPLE_LEGIT_CLAIM = {
  incident_severity: 'Minor Damage',
  incident_type: 'Multi-vehicle Collision',
  collision_type: 'Rear Collision',
  number_of_vehicles_involved: 2,
  incident_hour_of_the_day: 14,
  authorities_contacted: 'Police',
  police_report_available: 'YES',
  property_damage: 'NO',
  bodily_injuries: 0,
  witnesses: 2,
  total_claim_amount: 18000,
  vehicle_claim: 14000,
  property_claim: 2000,
  injury_claim: 2000,
  policy_deductable: 1000,
  policy_annual_premium: 1200,
  insured_sex: 'MALE',
  age: 38,
  insured_relationship: 'husband',
  auto_make: 'Toyota',
  auto_model: 'Camry',
  auto_year: 2012
};

const SAMPLE_FRAUD_CLAIM = {
  incident_severity: 'Major Damage',
  incident_type: 'Single Vehicle Collision',
  collision_type: 'Front Collision',
  number_of_vehicles_involved: 1,
  incident_hour_of_the_day: 3,
  authorities_contacted: 'Police',
  police_report_available: 'NO',
  property_damage: 'YES',
  bodily_injuries: 1,
  witnesses: 0,
  total_claim_amount: 76000,
  vehicle_claim: 56000,
  property_claim: 10000,
  injury_claim: 10000,
  policy_deductable: 1000,
  policy_annual_premium: 1350,
  insured_sex: 'MALE',
  age: 42,
  insured_relationship: 'husband',
  auto_make: 'Saab',
  auto_model: '9-3',
  auto_year: 2004
};

const PredictionForm = ({ onResult }) => {
  const [formData, setFormData] = useState(() => {
    const initialData = {};
    allFormFields.forEach(f => (initialData[f.name] = ''));
    return initialData;
  });

  const [loading, setLoading] = useState(false);
  const [wakingUp, setWakingUp] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const loadSample = (sample) => {
    setFormData(prev => ({ ...prev, ...sample }));
    setError(null);
  };

  const renderField = (field) => (
    <div key={field.name} className="prediction__group">
      <label htmlFor={field.name} className="prediction__label">
        {field.label}
      </label>
      {field.type === 'select' ? (
        <select
          id={field.name}
          name={field.name}
          value={formData[field.name]}
          onChange={handleChange}
          className="prediction__select"
        >
          {field.options.map((opt, idx) => (
            <option key={idx} value={opt}>
              {opt === '' ? 'Select...' : opt}
            </option>
          ))}
        </select>
      ) : (
        <input
          type={field.type === 'number' ? 'number' : 'text'}
          id={field.name}
          name={field.name}
          value={formData[field.name]}
          onChange={handleChange}
          placeholder={field.placeholder}
          className="prediction__input"
          step="any"
        />
      )}
    </div>
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setWakingUp(false);

    // Essential fields validation
    if (!formData.incident_severity || !formData.total_claim_amount) {
      setError('Please fill at least Incident Severity and Total Claim Amount before submitting.');
      return;
    }

    const payload = {};
    Object.keys(formData).forEach((k) => {
      if (formData[k] !== '') {
        const fieldSpec = allFormFields.find(f => f.name === k);
        payload[k] = fieldSpec?.type === 'number' ? Number(formData[k]) : formData[k];
      }
    });

    setLoading(true);
    try {
      const response = await predictFraud(payload, () => setWakingUp(true));
      onResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message || err.message || 'An error occurred during prediction.'
      );
    } finally {
      setLoading(false);
      setWakingUp(false);
    }
  };

  return (
    <section id="prediction" className="prediction">
      <div className="prediction__header">
        <div className="prediction__badge">
          <HiSparkles /> ExtraTrees Sampling ML Engine
        </div>
        <h2>Predict Vehicle Insurance Fraud</h2>
        <p>Screen vehicle claims in real-time. Load a realistic test preset or customize the claim variables below.</p>
        
        {/* Quick-fill sample buttons for instant testing */}
        <div className="prediction__presets">
          <span className="prediction__presets-label">
            <FaMagic /> Quick Presets:
          </span>
          <button
            type="button"
            className="prediction__preset-btn prediction__preset-btn--safe"
            onClick={() => loadSample(SAMPLE_LEGIT_CLAIM)}
          >
            Load Low-Risk Claim
          </button>
          <button
            type="button"
            className="prediction__preset-btn prediction__preset-btn--fraud"
            onClick={() => loadSample(SAMPLE_FRAUD_CLAIM)}
          >
            Load High-Risk Claim
          </button>
        </div>
      </div>

      <div className="prediction__card">
        <form className="prediction__form" onSubmit={handleSubmit}>
          {/* Section 1: Incident & Collision */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaCarCrash /> Incident & Collision Dynamics
            </h3>
            <div className="prediction__section-grid">
              {incidentFields.map(renderField)}
            </div>
          </div>

          {/* Section 2: Claim Financials */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaFileInvoiceDollar /> Claim Financial Breakdown
            </h3>
            <div className="prediction__section-grid">
              {claimFinancialFields.map(renderField)}
            </div>
          </div>

          {/* Section 3: Policyholder & Vehicle */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaUserTie /> Policyholder & Vehicle Information
            </h3>
            <div className="prediction__section-grid">
              {policyholderFields.map(renderField)}
            </div>
          </div>

          {wakingUp && (
            <div className="prediction__waking">
              <FaSpinner className="spinner" />
              <span>
                Our server is waking up from sleep (this happens on free hosting after
                inactivity). It can take up to a minute — your claim is being processed,
                please keep this page open.
              </span>
            </div>
          )}

          {error && (
            <div className="prediction__error">
              {error}
            </div>
          )}

          <div className="prediction__submit-wrapper">
            <button
              type="submit"
              className="prediction__submit"
              disabled={loading}
            >
              {loading ? (
                <>
                  <FaSpinner className="spinner" />
                  {wakingUp ? 'Waking Up Server...' : 'Analyzing Claim Risk...'}
                </>
              ) : (
                <>
                  <HiSparkles /> Analyze Claim Risk
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
};

export default PredictionForm;