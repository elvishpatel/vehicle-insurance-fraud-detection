import React, { useState } from 'react';
import { FaSpinner, FaUserTie, FaFileInvoiceDollar, FaCar } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { predictFraud } from '../services/api';
import './PredictionForm.css';

// Field definitions mirror the input schema of the served model.
// The form only asks for the 15 fields that carry the predictive signal;
// the remaining model inputs are auto-filled with typical values server-side.

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const policyFields = [
  { name: 'Month', label: 'Month of Policy', type: 'select', options: ['', ...MONTHS] },
  { name: 'WeekOfMonth', label: 'Week of Month (Policy)', placeholder: 'e.g. 3', type: 'number' },
  { name: 'Age', label: 'Age of Policy Holder', placeholder: 'e.g. 35', type: 'number' },
  { name: 'RepNumber', label: 'Agent / Representative Number', placeholder: 'e.g. 12', type: 'number' },
  { name: 'Year', label: 'Year of Claim', placeholder: 'e.g. 1996', type: 'number' },
  { name: 'BasePolicy', label: 'Base Policy', type: 'select', options: ['', 'All Perils', 'Collision', 'Liability'] },
  { name: 'PolicyType', label: 'Policy Type', type: 'select', options: ['', 'Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability', 'Sport - All Perils', 'Sport - Collision', 'Sport - Liability', 'Utility - All Perils', 'Utility - Collision', 'Utility - Liability'] },
];

const vehicleFields = [
  { name: 'VehiclePrice', label: 'Vehicle Price Band', type: 'select', options: ['', 'less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'] },
  { name: 'AgeOfVehicle', label: 'Age of Vehicle', type: 'select', options: ['', 'new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'] },
  { name: 'Deductible', label: 'Policy Deductible ($)', placeholder: 'e.g. 400', type: 'number' },
];

const claimFields = [
  { name: 'MonthClaimed', label: 'Month Claimed', type: 'select', options: ['', ...MONTHS, '0'] },
  { name: 'DayOfWeekClaimed', label: 'Day of Week Claimed', type: 'select', options: ['', ...WEEKDAYS, '0'] },
  { name: 'PastNumberOfClaims', label: 'Past Number of Claims', type: 'select', options: ['', 'none', '1', '2 to 4', 'more than 4'] },
  { name: 'Fault', label: 'Fault', type: 'select', options: ['', 'Policy Holder', 'Third Party'] },
  { name: 'AddressChange_Claim', label: 'Address Change Before Claim', type: 'select', options: ['', 'no change', 'under 6 months', '1 year', '2 to 3 years', '4 to 8 years'] },
];

const allFormFields = [...policyFields, ...vehicleFields, ...claimFields];

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
    setWakingUp(false);

    const isEmpty = Object.values(formData).some((value) => value === '');
    if (isEmpty) {
      setError('Please fill in all the fields before submitting.');
      return;
    }

    // Numeric fields are sent as numbers; categorical fields stay as strings.
    const payload = { ...formData };
    allFormFields
      .filter((f) => f.type === 'number')
      .forEach((f) => {
        payload[f.name] = Number(payload[f.name]);
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
          <HiSparkles /> Real-Time ML Engine
        </div>
        <h2>Predict Insurance Fraud</h2>
        <p>We only ask for the details that actually drive the prediction — everything else is auto-filled with typical values by our AI engine.</p>
      </div>

      <div className="prediction__card">
        <form className="prediction__form" onSubmit={handleSubmit}>
          {/* Section 1: Policy & Driver */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaUserTie /> Policy & Driver Details
            </h3>
            <div className="prediction__section-grid">
              {policyFields.map(renderField)}
            </div>
          </div>

          {/* Section 2: Vehicle */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaCar /> Vehicle Specifications
            </h3>
            <div className="prediction__section-grid">
              {vehicleFields.map(renderField)}
            </div>
          </div>

          {/* Section 3: Claim & Incident */}
          <div className="prediction__section">
            <h3 className="prediction__section-title">
              <FaFileInvoiceDollar /> Claim & Incident Details
            </h3>
            <div className="prediction__section-grid">
              {claimFields.map(renderField)}
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
                  {wakingUp ? 'Waking Up Server...' : 'Analyzing Claim...'}
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
