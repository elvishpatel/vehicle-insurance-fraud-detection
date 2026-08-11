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
import { FaChartLine } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './ModelAnalytics.css';

const dashboardTrendData = [
  { month: 'Jan', claims: 50 },
  { month: 'Feb', claims: 130 },
  { month: 'Mar', claims: 225 },
  { month: 'Apr', claims: 310 },
  { month: 'May', claims: 230 },
  { month: 'Jun', claims: 280 },
  { month: 'Jul', claims: 435 },
  { month: 'Aug', claims: 390 },
  { month: 'Sep', claims: 460 },
  { month: 'Oct', claims: 410 },
  { month: 'Nov', claims: 480 },
  { month: 'Dec', claims: 520 },
];

// Custom Tooltip for Dashboard Trend
const CustomTrendTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="dash__tooltip">
        <div className="dash__tooltip-label">{label}</div>
        <div className="dash__tooltip-val">Fraud Claims: <strong>{payload[0].value}</strong></div>
      </div>
    );
  }
  return null;
};

const ModelAnalytics = () => {
  return (
    <section id="analytics" className="analytics">
      <div className="analytics__container">
        {/* Mac Window Fraud Prediction Chart */}
        <div className="dash__mac-window">
          {/* Mac Window Title Bar */}
          <div className="dash__mac-bar">
            <div className="dash__mac-dots">
              <span className="dash__mac-dot dash__mac-dot--red"></span>
              <span className="dash__mac-dot dash__mac-dot--yellow"></span>
              <span className="dash__mac-dot dash__mac-dot--green"></span>
            </div>
            <div className="dash__mac-title">Fraud Prediction Dashboard</div>
            <div className="dash__mac-badge"><FaChartLine /> Live Spline Curve</div>
          </div>

          {/* Mac Window Body: Spline Curve Plot */}
          <div className="dash__mac-body">
            <ResponsiveContainer width="100%" height={360}>
              <AreaChart data={dashboardTrendData} margin={{ top: 25, right: 30, left: 0, bottom: 10 }}>
                <defs>
                  <linearGradient id="purpleSpline" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.6} />
                    <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
                <XAxis dataKey="month" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 13, fontWeight: 600 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 13 }} domain={[0, 550]} label={{ value: 'Claims', angle: -90, position: 'insideLeft', fill: '#94a3b8', offset: 15 }} />
                <Tooltip content={<CustomTrendTooltip />} />
                <Area
                  type="monotone"
                  dataKey="claims"
                  stroke="#818cf8"
                  strokeWidth={4}
                  fillOpacity={1}
                  fill="url(#purpleSpline)"
                  activeDot={{ r: 8, fill: '#ffffff', stroke: '#818cf8', strokeWidth: 3 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelAnalytics;
