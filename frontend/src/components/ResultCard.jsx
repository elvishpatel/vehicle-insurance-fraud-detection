import React, { useEffect, useRef, useState } from 'react';
import { FaExclamationTriangle, FaCheckCircle, FaShieldAlt, FaPercentage } from 'react-icons/fa';
import './ResultCard.css';

const ResultCard = ({ result }) => {
  const [progressWidth, setProgressWidth] = useState(0);
  const resultRef = useRef(null);

  useEffect(() => {
    if (result) {
      if (resultRef.current) {
        resultRef.current.scrollIntoView({ behavior: 'smooth' });
      }
      
      const timer = setTimeout(() => {
        const percentage = Math.round(result.probability * 100);
        setProgressWidth(percentage);
      }, 100);
      
      return () => clearTimeout(timer);
    }
  }, [result]);

  if (!result) return null;

  const isFraud = result.prediction === 'Fraud';
  const percentage = Math.round(result.probability * 100);

  const handleAnalyzeAnother = () => {
    const formElement = document.getElementById('prediction');
    if (formElement) {
      formElement.scrollIntoView({ behavior: 'smooth' });
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <section className="result" id="result" ref={resultRef}>
      <div className={`result__card ${isFraud ? 'result__card--fraud' : 'result__card--safe'}`}>
        <div className="result__icon">
          {isFraud ? <FaExclamationTriangle /> : <FaCheckCircle />}
        </div>
        
        <div className="result__badge">
          {isFraud ? 'High Risk Claim' : 'Low Risk Claim'}
        </div>
        
        <h2 className="result__prediction">
          {isFraud ? 'Fraud Detected' : 'Claim is Legitimate'}
        </h2>
        
        <div className="result__probability">
          <div className="result__probability-label">
            {isFraud ? 'Fraud Probability' : 'Safety Score'}
          </div>
          <div className="result__probability-value">
            {percentage}%
          </div>
          <div className="result__progress">
            <div 
              className="result__progress-bar" 
              style={{ width: `${progressWidth}%` }}
            ></div>
          </div>
        </div>
        
        <p className="result__description">
          {isFraud 
            ? 'This claim exhibits multiple high-risk indicators commonly associated with fraudulent activities. Further manual investigation is strongly recommended.'
            : 'This claim follows typical patterns of legitimate insurance claims. No significant risk indicators were detected by our AI systems.'
          }
        </p>
        
        <div className="result__footer">
          <FaShieldAlt />
          <span>FraudShield AI Analysis</span>
        </div>
        
        <button className="result__again-btn" onClick={handleAnalyzeAnother}>
          Analyze Another Claim
        </button>
      </div>
    </section>
  );
};

export default ResultCard;
