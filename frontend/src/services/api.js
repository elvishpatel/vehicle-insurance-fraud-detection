import axios from 'axios';

// Create axios instance with dynamic base URL (supports Vercel/Netlify env vars)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const API = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 15000, // 15 second timeout for production cold starts
});

/**
 * Send claim data to Flask API for fraud prediction
 * @param {Object} claimData - The insurance claim form data
 * @returns {Promise} - API response with prediction and probability
 */
export const predictFraud = async (claimData) => {
  try {
    const response = await API.post('/predict', claimData);
    return response;
  } catch (error) {
    if (error.code === 'ECONNABORTED') {
      throw new Error('Request timed out. Please check if the Flask server is running.');
    }
    if (!error.response) {
      throw new Error('Network error. Please ensure the Flask backend is running on http://localhost:5000');
    }
    throw new Error(error.response?.data?.message || 'Prediction failed. Please try again.');
  }
};

export default API;
