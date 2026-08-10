import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';
import { FaChartBar, FaChartArea, FaBezierCurve, FaBrain, FaLayerGroup } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './ModelAnalytics.css';

// 1. Feature Importance Data for Bar Chart
const featureData = [
  { name: 'Income', full: 'Annual Income', importance: 73.15, category: 'Primary' },
  { name: 'Days Open', full: 'Days Claim Open', importance: 4.48, category: 'Lifecycle' },
  { name: 'Safety', full: 'Safety Rating', importance: 3.25, category: 'Risk' },
  { name: 'Injury Claim', full: 'Injury Claim Amount', importance: 3.08, category: 'Severity' },
  { name: 'Driver Age', full: 'Age of Driver', importance: 2.48, category: 'Demographic' },
  { name: 'Vehicle Age', full: 'Age of Vehicle', importance: 1.95, category: 'Asset' },
  { name: 'Form Defects', full: 'Form Defects Count', importance: 1.68, category: 'Audit' },
  { name: 'Liability %', full: 'Liability Percentage', importance: 1.56, category: 'Fault' },
  { name: 'Total Claim', full: 'Total Claim Amount', importance: 1.13, category: 'Severity' },
  { name: 'Vehicle Price', full: 'Vehicle Price', importance: 0.69, category: 'Asset' },
];

// 2. ROC / Model Learning Curve Data for Area Chart
const rocData = [
  { threshold: '0.0', tpr: 0, fpr: 0, accuracy: 50.0 },
  { threshold: '0.1', tpr: 0.22, fpr: 0.02, accuracy: 62.4 },
  { threshold: '0.2', tpr: 0.45, fpr: 0.05, accuracy: 71.0 },
  { threshold: '0.3', tpr: 0.68, fpr: 0.09, accuracy: 76.5 },
  { threshold: '0.4', tpr: 0.81, fpr: 0.14, accuracy: 78.1 },
  { threshold: '0.5', tpr: 0.88, fpr: 0.18, accuracy: 78.1 },
  { threshold: '0.6', tpr: 0.93, fpr: 0.25, accuracy: 77.2 },
  { threshold: '0.7', tpr: 0.96, fpr: 0.35, accuracy: 74.8 },
  { threshold: '0.8', tpr: 0.98, fpr: 0.52, accuracy: 70.3 },
  { threshold: '0.9', tpr: 0.99, fpr: 0.74, accuracy: 63.5 },
  { threshold: '1.0', tpr: 1.0, fpr: 1.0, accuracy: 50.0 },
];

// 3. Risk Vector Radar Data
const radarData = [
  { subject: 'Income Anomaly', FraudRisk: 95, NormalClaim: 20 },
  { subject: 'Injury Claim Ratio', FraudRisk: 82, NormalClaim: 35 },
  { subject: 'Claim Days Open', FraudRisk: 75, NormalClaim: 40 },
  { subject: 'Liability %', FraudRisk: 68, NormalClaim: 45 },
  { subject: 'Vehicle Age Risk', FraudRisk: 60, NormalClaim: 30 },
  { subject: 'Past Claim History', FraudRisk: 85, NormalClaim: 25 },
];

const barColors = ['#818cf8', '#a855f7', '#38bdf8', '#c084fc', '#818cf8', '#34d399', '#f59e0b', '#6366f1', '#a855f7', '#38bdf8'];

// Custom Tooltip for Bar Chart
const CustomBarTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="analytics__tooltip">
        <div className="analytics__tooltip-title">{data.full}</div>
        <div className="analytics__tooltip-value">Importance: <strong>{data.importance}%</strong></div>
        <div className="analytics__tooltip-sub">Category: {data.category}</div>
      </div>
    );
  }
  return null;
};

// Custom Tooltip for ROC Curve
const CustomRocTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="analytics__tooltip">
        <div className="analytics__tooltip-title">Threshold: {data.threshold}</div>
        <div className="analytics__tooltip-value" style={{ color: '#818cf8' }}>
          Accuracy: <strong>{data.accuracy}%</strong>
        </div>
        <div className="analytics__tooltip-sub">
          True Positive Rate: {(data.tpr * 100).toFixed(0)}% | False Positive: {(data.fpr * 100).toFixed(0)}%
        </div>
      </div>
    );
  }
  return null;
};

const ModelAnalytics = () => {
  const [graphType, setGraphType] = useState('bar');

  return (
    <section id="analytics" className="analytics">
      <div className="analytics__container">
        {/* Header */}
        <div className="analytics__header">
          <div className="analytics__badge">
            <HiSparkles /> Interactive AI Charting
          </div>
          <h2 className="analytics__title">Model Performance & Feature Plots</h2>
          <p className="analytics__subtitle">
            Plotted Decision Tree analytics showing Gini feature importance curves, ROC accuracy trade-offs, and risk radar vectors.
          </p>
        </div>

        {/* Graph Switcher Controls */}
        <div className="analytics__tabs">
          <button
            className={`analytics__tab ${graphType === 'bar' ? 'analytics__tab--active' : ''}`}
            onClick={() => setGraphType('bar')}
          >
            <FaChartBar /> Feature Importance Bar Chart
          </button>
          <button
            className={`analytics__tab ${graphType === 'roc' ? 'analytics__tab--active' : ''}`}
            onClick={() => setGraphType('roc')}
          >
            <FaChartArea /> ROC & Accuracy Curve
          </button>
          <button
            className={`analytics__tab ${graphType === 'radar' ? 'analytics__tab--active' : ''}`}
            onClick={() => setGraphType('radar')}
          >
            <FaBezierCurve /> Fraud Risk Radar
          </button>
        </div>

        {/* Main Chart Card */}
        <div className="analytics__card glass">
          {/* Chart Header */}
          <div className="analytics__card-header">
            <div>
              <h3 className="analytics__card-title">
                {graphType === 'bar' && 'Decision Tree Feature Weight Distribution'}
                {graphType === 'roc' && 'Receiver Operating Characteristic (ROC) & Accuracy Curve'}
                {graphType === 'radar' && 'Multidimensional Fraud Risk Profile Radar'}
              </h3>
              <p className="analytics__card-sub">
                {graphType === 'bar' && 'Plotted X/Y coordinate bar graph of Gini impurity reduction per feature.'}
                {graphType === 'roc' && 'True Positive Rate vs Threshold plotting 78.1% peak accuracy cutoff.'}
                {graphType === 'radar' && 'Polar radar comparison of Fraudulent Claims vs Legitimate Claims.'}
              </p>
            </div>
            <div className="analytics__pill">
              <FaLayerGroup /> Live Plotted Recharts
            </div>
          </div>

          {/* Plotted Graph Container */}
          <div className="analytics__chart-wrapper">
            {/* 1. Bar Chart Plot */}
            {graphType === 'bar' && (
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={featureData} margin={{ top: 20, right: 30, left: 10, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
                  <XAxis
                    dataKey="name"
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 12, fontWeight: 600 }}
                    angle={-25}
                    textAnchor="end"
                    interval={0}
                  />
                  <YAxis
                    stroke="#94a3b8"
                    tick={{ fill: '#94a3b8', fontSize: 12 }}
                    unit="%"
                    domain={[0, 80]}
                  />
                  <Tooltip content={<CustomBarTooltip />} cursor={{ fill: 'rgba(255, 255, 255, 0.04)' }} />
                  <Bar dataKey="importance" radius={[6, 6, 0, 0]}>
                    {featureData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={barColors[index % barColors.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}

            {/* 2. ROC / Accuracy Curve Plot */}
            {graphType === 'roc' && (
              <ResponsiveContainer width="100%" height={380}>
                <AreaChart data={rocData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                  <defs>
                    <linearGradient id="accuracyGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#6366f1" stopOpacity={0.8} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                    <linearGradient id="tprGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#a855f7" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#a855f7" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" />
                  <XAxis dataKey="threshold" stroke="#94a3b8" label={{ value: 'Decision Threshold', position: 'insideBottom', offset: -10, fill: '#94a3b8' }} />
                  <YAxis stroke="#94a3b8" unit="%" domain={[0, 100]} />
                  <Tooltip content={<CustomRocTooltip />} />
                  <Area type="monotone" dataKey="accuracy" name="Accuracy (%)" stroke="#6366f1" strokeWidth={3} fillOpacity={1} fill="url(#accuracyGradient)" />
                  <Area type="monotone" dataKey="tpr" name="True Positive Rate" stroke="#a855f7" strokeWidth={2} fillOpacity={1} fill="url(#tprGradient)" />
                </AreaChart>
              </ResponsiveContainer>
            )}

            {/* 3. Radar Chart Plot */}
            {graphType === 'radar' && (
              <ResponsiveContainer width="100%" height={380}>
                <RadarChart outerRadius={130} data={radarData}>
                  <PolarGrid stroke="rgba(255, 255, 255, 0.15)" />
                  <PolarAngleAxis dataKey="subject" stroke="#cbd5e1" tick={{ fill: '#cbd5e1', fontSize: 12, fontWeight: 600 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#94a3b8" />
                  <Radar name="Fraudulent Claim Profile" dataKey="FraudRisk" stroke="#ef4444" fill="#ef4444" fillOpacity={0.5} />
                  <Radar name="Legitimate Claim Profile" dataKey="NormalClaim" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                  <Legend wrapperStyle={{ paddingTop: 10 }} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Model Summary Footer */}
          <div className="analytics__footer-stats">
            <div className="analytics__stat-box">
              <span className="analytics__stat-num">78.1%</span>
              <span className="analytics__stat-lbl">Model Accuracy</span>
            </div>
            <div className="analytics__stat-box">
              <span className="analytics__stat-num">73.15%</span>
              <span className="analytics__stat-lbl">Income Feature Weight</span>
            </div>
            <div className="analytics__stat-box">
              <span className="analytics__stat-num">11,716</span>
              <span className="analytics__stat-lbl">Trained Claims</span>
            </div>
            <div className="analytics__stat-box">
              <span className="analytics__stat-num">&lt; 20ms</span>
              <span className="analytics__stat-lbl">Plot Traversal Speed</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelAnalytics;
