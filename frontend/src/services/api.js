import axios from 'axios';

// Create axios instance with dynamic base URL (supports Vercel/Netlify env vars)
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';

const API = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// The backend is hosted on a free tier that sleeps after inactivity. The first
// request then has to wait while the server boots (~30-60s), which is longer
// than a normal API timeout. When that happens we retry once with a generous
// window instead of failing, and tell the UI so it can show a friendly
// "server waking up" message.
const FAST_TIMEOUT_MS = 15000;
const WAKE_TIMEOUT_MS = 120000;

/**
 * Send claim data to Flask API for fraud prediction
 * @param {Object} claimData - The insurance claim form data
 * @param {Function} [onWakingUp] - Called when a sleeping server is detected
 *   and the patient retry has started, so the UI can inform the user.
 * @returns {Promise} - API response with prediction and probability
 */
export const predictFraud = async (claimData, onWakingUp) => {
  const request = (timeout) => API.post('/predict', claimData, { timeout });

  try {
    return await request(FAST_TIMEOUT_MS);
  } catch (firstError) {
    const isServerWaking =
      firstError.code === 'ECONNABORTED' ||        // timed out while server boots
      !firstError.response ||                      // network error / connection refused
      firstError.response?.status >= 502;          // bad gateway / service unavailable

    if (!isServerWaking) {
      throw new Error(
        firstError.response?.data?.message || 'Prediction failed. Please try again.'
      );
    }

    if (typeof onWakingUp === 'function') onWakingUp();

    try {
      return await request(WAKE_TIMEOUT_MS);
    } catch (secondError) {
      if (secondError.code === 'ECONNABORTED' || !secondError.response) {
        throw new Error(
          'The server is still waking up from sleep. Please wait about a minute, then try again — your data is safe.'
        );
      }
      throw new Error(
        secondError.response?.data?.message || 'Prediction failed. Please try again.'
      );
    }
  }
};

export default API;
