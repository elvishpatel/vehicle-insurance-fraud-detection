import React, { useState } from 'react';
import { FaSpinner, FaUserTie, FaFileInvoiceDollar, FaCar } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import { predictFraud } from '../services/api';
import './PredictionForm.css';

// Field definitions mirror the input schema of the served model exactly
// (see CATEGORY_FIELDS / NUMERIC_FIELDS in the Flask app.py).

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

const policyFields = [
  { name: 'Month', label: 'Month of Policy', type: 'select', options: ['', ...MONTHS] },
  { name: 'WeekOfMonth', label: 'Week of Month (Policy)', placeholder: 'e.g. 3', type: 'number' },
  { name: 'DayOfWeek', label: 'Day of Week (Policy)', type: 'select', options: ['', ...WEEKDAYS] },
  { name: 'Sex', label: 'Gender', type: 'select', options: ['', 'Male', 'Female'] },
  { name: 'MaritalStatus', label: 'Marital Status', type: 'select', options: ['', 'Single', 'Married', 'Divorced', 'Widow'] },
  { name: 'Age', label: 'Age of Policy Holder', placeholder: 'e.g. 35', type: 'number' },
  { name: 'AgeOfPolicyHolder', label: 'Age Band of Holder', type: 'select', options: ['', '16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 40', '41 to 50', '51 to 65', 'over 65'] },
  { name: 'DriverRating', label: 'Driver Rating (1-4)', placeholder: 'e.g. 1', type: 'number' },
  { name: 'RepNumber', label: 'Agent / Representative Number', placeholder: 'e.g. 12', type: 'number' },
  { name: 'Year', label: 'Year of Claim', placeholder: 'e.g. 1996', type: 'number' },
  { name: 'BasePolicy', label: 'Base Policy', type: 'select', options: ['', 'All Perils', 'Collision', 'Liability'] },
  { name: 'PolicyType', label: 'Policy Type', type: 'select', options: ['', 'Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability', 'Sport - All Perils', 'Sport - Collision', 'Sport - Liability', 'Utility - All Perils', 'Utility - Collision', 'Utility - Liability'] },
];

const vehicleFields = [
  { name: 'Make', label: 'Vehicle Make', type: 'select', options: ['', 'Accura', 'BMW', 'Chevrolet', 'Dodge', 'Ferrari', 'Ford', 'Honda', 'Jaguar', 'Lexus', 'Mazda', 'Mecedes', 'Mercury', 'Nisson', 'Pontiac', 'Porche', 'Saab', 'Saturn', 'Toyota', 'VW'] },
  { name: 'VehicleCategory', label: 'Vehicle Category', type: 'select', options: ['', 'Sedan', 'Sport', 'Utility'] },
  { name: 'VehiclePrice', label: 'Vehicle Price Band', type: 'select', options: ['', 'less than 20000', '20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'more than 69000'] },
  { name: 'AgeOfVehicle', label: 'Age of Vehicle', type: 'select', options: ['', 'new', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', 'more than 7'] },
  { name: 'NumberOfCars', label: 'Number of Cars', type: 'select', options: ['', '1 vehicle', '2 vehicles', '3 to 4', '5 to 8', 'more than 8'] },
  { name: 'Deductible', label: 'Policy Deductible ($)', placeholder: 'e.g. 400', type: 'number' },
];

const claimFields = [
  { name: 'AccidentArea', label: 'Accident Area', type: 'select', options: ['', 'Urban', 'Rural'] },
  { name: 'MonthClaimed', label: 'Month Claimed', type: 'select', options: ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '0'] },
  { name: 'DayOfWeekClaimed', label: 'Day of Week Claimed', type: 'select', options: ['', ...WEEKDAYS, '0'] },
  { name: 'WeekOfMonthClaimed', label: 'Week of Month Claimed', placeholder: 'e.g. 2', type: 'number' },
  { name: 'Days_Policy_Accident', label: 'Days: Policy to Accident', type: 'select', options: ['', 'none', '1 to 7', '8 to 15', '15 to 30', 'more than 30'] },
  { name: 'Days_Policy_Claim', label: 'Days: Policy to Claim', type: 'select', options: ['', 'none', '8 to 15', '15 to 30', 'more than 30'] },
  { name: 'PastNumberOfClaims', label: 'Past Number of Claims', type: 'select', options: ['', 'none', '1', '2 to 4', 'more than 4'] },
  { name: 'Fault', label: 'Fault', type: 'select', options: ['', 'Policy Holder', 'Third Party'] },
  { name: 'PoliceReportFiled', label: 'Police Report Filed', type: 'select', options: ['', 'Yes', 'No'] },
  { name: 'WitnessPresent', label: 'Witness Present', type: 'select', options: ['', 'Yes', 'No'] },
  { name: 'AgentType', label: 'Agent Type', type: 'select', options: ['', 'External', 'Internal'] },
  { name: 'NumberOfSuppliments', label: 'Number of Supplements', type: 'select', options: ['', 'none', '1 to 2', '3 to 5', 'more than 5'] },
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

    // Numeric fields are sent as numbers; categorical fields stay as strings.
    const payload = { ...formData };
    allFormFields
      .filter((f) => f.type === 'number')
      .forEach((f) => {
        payload[f.name] = Number(payload[f.name]);
      });

    setLoading(true);
    try {
      const response = await predictFraud(payload);
      onResult(response.data);
    } catch (err) {
      setError(
        err.response?.data?.message || err.message || 'An error occurred during prediction.'
      );
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
        <p>Enter the claim details below exactly as recorded on the policy. Encoding is applied automatically by our AI engine.</p>
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
