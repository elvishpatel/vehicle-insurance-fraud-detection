import React, { useState } from 'react';
import { FaSpinner, FaUserTie, FaFileInvoiceDollar, FaCar } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { predictFraud } from '../services/api';
import './PredictionForm.css';

const driverPolicyFields = [
  { name: 'age_of_driver', label: 'Age of Driver (Years)', placeholder: 'e.g. 35', type: 'number' },
  { name: 'annual_income', label: 'Annual Income (₹)', placeholder: 'e.g. ₹60,000', type: 'number' },
  { name: 'gender', label: 'Gender', type: 'select', options: ['', 'Male', 'Female'] },
  { name: 'safety_rating', label: 'Safety Rating (0-100)', placeholder: 'e.g. 85', type: 'number' },
  { name: 'property_status', label: 'Property Status', type: 'select', options: ['', 'Own', 'Rent'] },
  { name: 'address_change', label: 'Recent Address Change', type: 'select', options: ['', 'Yes', 'No'] },
  { name: 'annual_premium', label: 'Annual Premium (₹)', placeholder: 'e.g. ₹1,200', type: 'number' },
  { name: 'policy_deductible', label: 'Policy Deductible (₹)', placeholder: 'e.g. ₹500', type: 'number' },
];

const claimFields = [
  { name: 'total_claim', label: 'Total Claim Amount (₹)', placeholder: 'e.g. ₹15,000', type: 'number' },
  { name: 'injury_claim', label: 'Injury Claim Amount (₹)', placeholder: 'e.g. ₹3,000', type: 'number' },
  { name: 'days_open', label: 'Days Claim Open', placeholder: 'e.g. 30', type: 'number' },
  { name: 'police_report', label: 'Police Report Filed', type: 'select', options: ['', 'Yes', 'No'] },
  { name: 'accident_site', label: 'Accident Site', type: 'select', options: ['', 'Highway', 'Local', 'Parking Lot'] },
  { name: 'liab_prct', label: 'Liability Percentage (%)', placeholder: 'e.g. 25', type: 'number' },
  { name: 'past_num_of_claims', label: 'Past Number of Claims', placeholder: 'e.g. 2', type: 'number' },
  { name: 'form_defects', label: 'Form Defects Count', placeholder: 'e.g. 0', type: 'number' },
];

const vehicleFields = [
  { name: 'age_of_vehicle', label: 'Vehicle Age (Years)', placeholder: 'e.g. 4', type: 'number' },
  { name: 'vehicle_price', label: 'Vehicle Price (₹)', placeholder: 'e.g. ₹25,000', type: 'number' },
  { name: 'vehicle_category', label: 'Vehicle Category', type: 'select', options: ['', 'Compact', 'Medium', 'Large'] },
];

const allFormFields = [...driverPolicyFields, ...claimFields, ...vehicleFields];

const PredictionForm = ({ onResult }) => {
  const [formData, setFormData] = useState(() => {
    const initialData = {};
    allFormFields.forEach(f => (initialData[f.name] = ''));
    return initialData;
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const renderField = (field) => (
    <div key={field.name} className="prediction__group">
      <label htmlFor={field.name} className="prediction__label">
        {field.label}
      </label>
      {field.type === 'number' ? (
        <input
          type="number"
          id={field.name}
          name={field.name}
          value={formData[field.name]}
          onChange={handleChange}
          placeholder={field.placeholder}
          className="prediction__input"
          step="any"
        />
      ) : (
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
      )}
    </div>
  );

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    const isEmpty = Object.values(formData).some((value) => value === '');
    if (isEmpty) {
      setError('Please fill in all the fields before submitting.');
      return;
    }

    setLoading(true);
    try {
      const response = await predictFraud(formData);
      onResult(response.data);
    } catch (err) {
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="prediction" className="prediction">
      <div className="prediction__header">
        <div className="prediction__badge">
          <HiSparkles /> Real-Time ML Engine
        </div>
        <h2>Predict Insurance Fraud</h2>
        <p>Enter the claim details below in standard units. Automatic scaling is applied by our AI engine.</p>
      </div>

      <div className="prediction__card">
        <form className="prediction__form" onSubmit={handleSubmit}>
          {/* Section 1: Driver & Policy */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaUserTie /> Driver & Policy Details
            </h3>
            <div className="prediction__section-grid">
              {driverPolicyFields.map(renderField)}
            </div>
          </div>

          {/* Section 2: Claim & Loss Details */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaFileInvoiceDollar /> Claim & Incident Details
            </h3>
            <div className="prediction__section-grid">
              {claimFields.map(renderField)}
            </div>
          </div>

          {/* Section 3: Vehicle Specs */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaCar /> Vehicle Specifications
            </h3>
            <div className="prediction__section-grid">
              {vehicleFields.map(renderField)}
            </div>
          </div>

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
                  <FaSpinner className="spinner" /> Analyzing Claim...
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
