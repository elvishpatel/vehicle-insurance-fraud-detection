import React, { useState } from 'react';
import { FaSpinner } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { predictFraud } from '../services/api';
import './PredictionForm.css';

const numericFields = [
  { name: 'age_of_driver', label: 'Age of Driver (Years)', placeholder: 'e.g. 35' },
  { name: 'safety_rating', label: 'Safety Rating (0-100)', placeholder: 'e.g. 85' },
  { name: 'annual_income', label: 'Annual Income ($)', placeholder: 'e.g. 60000' },
  { name: 'past_num_of_claims', label: 'Past Number of Claims', placeholder: 'e.g. 2' },
  { name: 'liab_prct', label: 'Liability Percentage (%)', placeholder: 'e.g. 25' },
  { name: 'age_of_vehicle', label: 'Vehicle Age (Years)', placeholder: 'e.g. 4' },
  { name: 'vehicle_price', label: 'Vehicle Price ($)', placeholder: 'e.g. 25000' },
  { name: 'total_claim', label: 'Total Claim Amount ($)', placeholder: 'e.g. 15000' },
  { name: 'injury_claim', label: 'Injury Claim Amount ($)', placeholder: 'e.g. 3000' },
  { name: 'policy_deductible', label: 'Policy Deductible ($)', placeholder: 'e.g. 500' },
  { name: 'annual_premium', label: 'Annual Premium ($)', placeholder: 'e.g. 1200' },
  { name: 'days_open', label: 'Days Claim Open', placeholder: 'e.g. 30' },
  { name: 'form_defects', label: 'Form Defects Count', placeholder: 'e.g. 0' },
];

const dropdownFields = [
  { name: 'high_education', label: 'Higher Education Degree', options: ['', 'Yes', 'No'] },
  { name: 'address_change', label: 'Recent Address Change', options: ['', 'Yes', 'No'] },
  { name: 'police_report', label: 'Police Report Filed', options: ['', 'Yes', 'No'] },
  { name: 'gender', label: 'Gender', options: ['', 'Male', 'Female'] },
  { name: 'marital_status', label: 'Marital Status', options: ['', 'Single', 'Married', 'Other'] },
  { name: 'property_status', label: 'Property Status', options: ['', 'Own', 'Rent'] },
  { name: 'claim_day_of_week', label: 'Claim Day of Week', options: ['', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] },
  { name: 'accident_site', label: 'Accident Site', options: ['', 'Highway', 'Local', 'Parking Lot'] },
  { name: 'witness_present', label: 'Witness Present', options: ['', 'Yes', 'No', 'Unknown'] },
  { name: 'channel', label: 'Sales Channel', options: ['', 'Broker', 'Online', 'Phone'] },
  { name: 'vehicle_category', label: 'Vehicle Category', options: ['', 'Compact', 'Medium', 'Large'] },
  { name: 'vehicle_color', label: 'Vehicle Color', options: ['', 'Black', 'Blue', 'Gray', 'Red', 'Silver', 'White', 'Other'] },
];

const PredictionForm = ({ onResult }) => {
  const [formData, setFormData] = useState(() => {
    const initialData = {};
    numericFields.forEach(f => (initialData[f.name] = ''));
    dropdownFields.forEach(f => (initialData[f.name] = ''));
    return initialData;
  });
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);

    // Check required fields
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
        <h2>Predict Insurance Fraud</h2>
        <p>Enter the claim details below in standard units. Automatic scaling is applied by our AI engine.</p>
      </div>

      <div className="prediction__card">
        <form className="prediction__form" onSubmit={handleSubmit}>
          {numericFields.map((field) => (
            <div key={field.name} className="prediction__group">
              <label htmlFor={field.name} className="prediction__label">
                {field.label}
              </label>
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
            </div>
          ))}

          {dropdownFields.map((field) => (
            <div key={field.name} className="prediction__group">
              <label htmlFor={field.name} className="prediction__label">
                {field.label}
              </label>
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
            </div>
          ))}

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
                  <HiSparkles /> Analyze Claim
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
