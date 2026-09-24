import React from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip
} from 'recharts';
import { FaChartLine, FaDatabase, FaSearchDollar, FaBullseye, FaPercentage, FaBalanceScale, FaChartBar, FaCrosshairs } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './ModelAnalytics.css';

// Real monthly claim volumes from the 15,420-claim training dataset
// (1994-1996 US automobile insurance claims).
const monthlyData = [
  { month: 'Jan', legit: 1324, fraud: 87 },
  { month: 'Feb', legit: 1184, fraud: 82 },
  { month: 'Mar', legit: 1258, fraud: 102 },
  { month: 'Apr', legit: 1200, fraud: 80 },
  { month: 'May', legit: 1273, fraud: 94 },
  { month: 'Jun', legit: 1241, fraud: 80 },
  { month: 'Jul', legit: 1197, fraud: 60 },
  { month: 'Aug', legit: 1043, fraud: 84 },
  { month: 'Sep', legit: 1164, fraud: 76 },
  { month: 'Oct', legit: 1235, fraud: 70 },
  { month: 'Nov', legit: 1155, fraud: 46 },
  { month: 'Dec', legit: 1223, fraud: 62 },
];

const stats = [
  { icon: FaBullseye, value: '82.50%', label: 'Accuracy' },
  { icon: FaCrosshairs, value: '62.50%', label: 'Precision' },
  { icon: FaSearchDollar, value: '71.43%', label: 'Recall' },
  { icon: FaChartBar, value: '66.67%', label: 'F1 Score' },
  { icon: FaPercentage, value: '84.17%', label: 'ROC-AUC' },
  { icon: FaBalanceScale, value: '78.76%', label: 'Balanced Accuracy' },
  { icon: FaChartLine, value: '61.24%', label: 'PR-AUC' },
  { icon: FaDatabase, value: '55.10%', label: 'MCC' },
];

// Custom Tooltip for the dashboard chart
const CustomTrendTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="dash__tooltip">
        <div className="dash__tooltip-label">{label}</div>
        {payload.map((entry) => (
          <div key={entry.dataKey} className="dash__tooltip-val">
            <span className="dash__tooltip-dot" style={{ background: entry.stroke || entry.fill }} />
            {entry.name}: <strong>{entry.value.toLocaleString()}</strong>
          </div>
        ))}
      </div>
    );
  }
  return null;
};

const ModelAnalytics = () => {
  return (
    <section id="analytics" className="analytics">
      <div className="analytics__container">
        <div className="analytics__header">
         
          <h2 className="analytics__title">Model Performance Metrics</h2>
          <p className="analytics__subtitle">
            Comprehensive evaluation metrics showing the ExtraTrees Sampling model's
            ability to detect fraudulent insurance claims with high accuracy and precision.
          </p>
        </div>

        {/* Key metrics */}
        <div className="dash__stats">
          {stats.map((stat) => (
            <div key={stat.label} className="dash__stat-card">
              <div className="dash__stat-icon">
                <stat.icon />
              </div>
              <div className="dash__stat-value">{stat.value}</div>
              <div className="dash__stat-label">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Mac Window Claims Chart */}
        <div className="dash__mac-window">
          <div className="dash__mac-bar">
            <div className="dash__mac-dots">
              <span className="dash__mac-dot dash__mac-dot--red"></span>
              <span className="dash__mac-dot dash__mac-dot--yellow"></span>
              <span className="dash__mac-dot dash__mac-dot--green"></span>
            </div>
            <div className="dash__mac-title">Monthly Claims: Legitimate vs Fraud</div>
            <div className="dash__mac-badge"><FaChartLine /> Training Data</div>
          </div>

          <div className="dash__mac-body">
            <ResponsiveContainer width="100%" height={360}>
              <AreaChart data={monthlyData} margin={{ top: 25, right: 30, left: 0, bottom: 10 }}>
                <defs>
                  <linearGradient id="legitFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6366f1" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
                <XAxis dataKey="month" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 13, fontWeight: 600 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 13 }} domain={[0, 1400]} label={{ value: 'Claims', angle: -90, position: 'insideLeft', fill: '#94a3b8', offset: 15 }} />
                <Tooltip content={<CustomTrendTooltip />} />
                <Area
                  type="monotone"
                  dataKey="legit"
                  name="Legitimate"
                  stroke="#818cf8"
                  strokeWidth={3}
                  fillOpacity={1}
                  fill="url(#legitFill)"
                  activeDot={{ r: 6, fill: '#ffffff', stroke: '#818cf8', strokeWidth: 3 }}
                />
                <Area
                  type="monotone"
                  dataKey="fraud"
                  name="Fraud"
                  stroke="#f87171"
                  strokeWidth={3}
                  fill="transparent"
                  activeDot={{ r: 6, fill: '#ffffff', stroke: '#f87171', strokeWidth: 3 }}
                />
              </AreaChart>
            </ResponsiveContainer>

            <div className="dash__legend">
              <span className="dash__legend-item">
                <span className="dash__legend-dot dash__legend-dot--legit"></span> Legitimate claims
              </span>
              <span className="dash__legend-item">
                <span className="dash__legend-dot dash__legend-dot--fraud"></span> Confirmed fraud
              </span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelAnalytics;
