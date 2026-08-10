import React, { useState } from 'react';
import { FaChartBar, FaBrain, FaLayerGroup, FaCheckCircle, FaInfoCircle } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './ModelAnalytics.css';

const featureImportances = [
  { name: 'Annual Income', percentage: 73.15, category: 'Primary Driver', color: '#818cf8', desc: 'Highest decision node weight in detecting anomaly signals.' },
  { name: 'Days Claim Open', percentage: 4.48, category: 'Claim Lifecycle', color: '#a855f7', desc: 'Extended resolution periods correlate with investigation complexity.' },
  { name: 'Safety Rating', percentage: 3.25, category: 'Risk Profile', color: '#38bdf8', desc: 'Vehicle and policyholder historical safety score impact.' },
  { name: 'Injury Claim Amount', percentage: 3.08, category: 'Loss Severity', color: '#c084fc', desc: 'Proportion of bodily injury claim vs overall claim magnitude.' },
  { name: 'Age of Driver', percentage: 2.48, category: 'Demographic', color: '#818cf8', desc: 'Driver experience and age bracket vulnerability metrics.' },
  { name: 'Age of Vehicle', percentage: 1.95, category: 'Asset Profile', color: '#34d399', desc: 'Vehicle depreciation and risk correlation index.' },
  { name: 'Form Defects', percentage: 1.68, category: 'Audit Check', color: '#f59e0b', desc: 'Inconsistencies or missing fields in filed document forms.' },
  { name: 'Liability %', percentage: 1.56, category: 'Fault Ratio', color: '#6366f1', desc: 'Determined liability split percentage in accident reports.' },
  { name: 'Total Claim Amount', percentage: 1.13, category: 'Loss Severity', color: '#a855f7', desc: 'Overall monetary value claimed in the incident.' },
  { name: 'Vehicle Price', percentage: 0.69, category: 'Asset Profile', color: '#38bdf8', desc: 'Market valuation of insured vehicle at claim time.' },
];

const ModelAnalytics = () => {
  const [activeTab, setActiveTab] = useState('features');
  const [hoveredFeature, setHoveredFeature] = useState(null);

  return (
    <section id="analytics" className="analytics">
      <div className="analytics__container">
        {/* Header */}
        <div className="analytics__header">
          <div className="analytics__badge">
            <HiSparkles /> Decision Tree Analytics
          </div>
          <h2 className="analytics__title">Model Performance & Insights</h2>
          <p className="analytics__subtitle">
            Visualizing the trained Decision Tree feature importances and dataset parameters behind FraudShield.
          </p>
        </div>

        {/* Navigation Tabs */}
        <div className="analytics__tabs">
          <button
            className={`analytics__tab ${activeTab === 'features' ? 'analytics__tab--active' : ''}`}
            onClick={() => setActiveTab('features')}
          >
            <FaChartBar /> Feature Importance Weights
          </button>
          <button
            className={`analytics__tab ${activeTab === 'metrics' ? 'analytics__tab--active' : ''}`}
            onClick={() => setActiveTab('metrics')}
          >
            <FaBrain /> Evaluation Metrics
          </button>
        </div>

        {/* Tab 1: Feature Importance Chart */}
        {activeTab === 'features' && (
          <div className="analytics__card glass">
            <div className="analytics__card-header">
              <div>
                <h3 className="analytics__card-title">Decision Node Importance Weights</h3>
                <p className="analytics__card-sub">Top 10 features sorted by Gini impurity reduction percentage in Decision Tree.</p>
              </div>
              <div className="analytics__pill">
                <FaLayerGroup /> 50 Input Features Evaluated
              </div>
            </div>

            <div className="analytics__chart">
              {featureImportances.map((item, index) => (
                <div
                  key={index}
                  className="analytics__bar-group"
                  onMouseEnter={() => setHoveredFeature(item)}
                  onMouseLeave={() => setHoveredFeature(null)}
                >
                  <div className="analytics__bar-info">
                    <span className="analytics__bar-name">{item.name}</span>
                    <span className="analytics__bar-value">{item.percentage.toFixed(2)}%</span>
                  </div>
                  <div className="analytics__bar-track">
                    <div
                      className="analytics__bar-fill"
                      style={{
                        width: `${Math.max(item.percentage, 2.5)}%`,
                        background: `linear-gradient(90deg, #6366f1 0%, ${item.color} 100%)`,
                      }}
                    >
                      <span className="analytics__bar-glow"></span>
                    </div>
                  </div>
                  <span className="analytics__bar-category">{item.category}</span>
                </div>
              ))}
            </div>

            {/* Hovered Feature Detail Banner */}
            <div className="analytics__insight-box">
              <FaInfoCircle className="analytics__insight-icon" />
              <div>
                <strong>
                  {hoveredFeature ? `${hoveredFeature.name} (${hoveredFeature.percentage}%)` : 'Hover over any bar'}
                </strong>
                <p>
                  {hoveredFeature
                    ? hoveredFeature.desc
                    : 'Annual Income is the primary root node split in the Decision Tree classifier, driving 73.15% of prediction weight.'}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Model Evaluation Metrics */}
        {activeTab === 'metrics' && (
          <div className="analytics__grid">
            <div className="analytics__metric-card glass">
              <div className="analytics__metric-icon" style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8' }}>
                <FaCheckCircle />
              </div>
              <div className="analytics__metric-num">78.1%</div>
              <div className="analytics__metric-label">Training & Validation Accuracy</div>
              <p className="analytics__metric-desc">Evaluated across 11,716 verified insurance claim records in dataset.</p>
            </div>

            <div className="analytics__metric-card glass">
              <div className="analytics__metric-icon" style={{ background: 'rgba(168, 85, 247, 0.15)', color: '#c084fc' }}>
                <FaBrain />
              </div>
              <div className="analytics__metric-num">50</div>
              <div className="analytics__metric-label">One-Hot Encoded Features</div>
              <p className="analytics__metric-desc">Includes scaled numeric ranges, binary indicators, and spatial claim attributes.</p>
            </div>

            <div className="analytics__metric-card glass">
              <div className="analytics__metric-icon" style={{ background: 'rgba(56, 189, 248, 0.15)', color: '#38bdf8' }}>
                <FaChartBar />
              </div>
              <div className="analytics__metric-num">&lt; 20ms</div>
              <div className="analytics__metric-label">Inference Execution Speed</div>
              <p className="analytics__metric-desc">Real-time Decision Tree traversal with zero latency bottlenecks.</p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
};

export default ModelAnalytics;
