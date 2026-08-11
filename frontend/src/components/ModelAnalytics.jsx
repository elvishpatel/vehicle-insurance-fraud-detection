import React, { useState } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar
} from 'recharts';
import { FaShieldAlt, FaExclamationTriangle, FaCheckCircle, FaCar, FaChartLine } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './ModelAnalytics.css';

// ── 1. Fraud vs Not Fraud Over Time (Image 1 Data) ─────────────
const monthlyComparisonData = [
  { month: 'Jan', NotFraud: 430, Fraud: 200 },
  { month: 'Feb', NotFraud: 490, Fraud: 140 },
  { month: 'Mar', NotFraud: 590, Fraud: 200 },
  { month: 'Apr', NotFraud: 460, Fraud: 155 },
  { month: 'May', NotFraud: 540, Fraud: 230 },
  { month: 'Jun', NotFraud: 690, Fraud: 265 },
  { month: 'Jul', NotFraud: 550, Fraud: 220 },
  { month: 'Aug', NotFraud: 690, Fraud: 330 },
  { month: 'Sep', NotFraud: 665, Fraud: 340 },
  { month: 'Oct', NotFraud: 545, Fraud: 225 },
  { month: 'Nov', NotFraud: 580, Fraud: 240 },
  { month: 'Dec', NotFraud: 685, Fraud: 365 },
];

// ── 2. Fraud Prediction Dashboard Trend (Image 2 & 3 Data) ──────
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

// ── 3. Claim Distribution Donut Data (Image 3 Top Right) ────────
const claimDistributionData = [
  { name: 'Not Fraud', value: 72.5, color: '#6366f1' },
  { name: 'Fraud', value: 27.5, color: '#ef4444' },
];

// ── 4. Fraud by Vehicle Category Pie Data (Image 4 Right) ──────
const vehicleCategoryData = [
  { name: 'Medium / SUV', value: 42, color: '#6366f1' },
  { name: 'Compact', value: 34, color: '#3b82f6' },
  { name: 'Large / Luxury', value: 24, color: '#10b981' },
];

// ── 5. Sparkline Data (Image 4 Cards) ───────────────────────────
const totalClaimsSparkline = [
  { v: 800 }, { v: 920 }, { v: 850 }, { v: 1050 }, { v: 1120 }, { v: 980 }, { v: 1200 }, { v: 1150 }
];
const fraudClaimsSparkline = [
  { v: 210 }, { v: 280 }, { v: 240 }, { v: 340 }, { v: 310 }, { v: 260 }, { v: 380 }, { v: 350 }
];
const accuracySparkline = [
  { v: 72 }, { v: 74 }, { v: 75 }, { v: 77 }, { v: 76 }, { v: 78 }, { v: 78.1 }, { v: 78.1 }
];

// ── 6. Feature Importance Bar Data ──────────────────────────────
const featureImportanceData = [
  { name: 'Annual Income', value: 73.15, fill: '#818cf8' },
  { name: 'Days Open', value: 4.48, fill: '#a855f7' },
  { name: 'Safety Rating', value: 3.25, fill: '#38bdf8' },
  { name: 'Injury Claim', value: 3.08, fill: '#c084fc' },
  { name: 'Driver Age', value: 2.48, fill: '#34d399' },
  { name: 'Vehicle Age', value: 1.95, fill: '#f59e0b' },
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
  const [activeThemeTab, setActiveThemeTab] = useState('dark');

  return (
    <section id="analytics" className="analytics">
      <div className="analytics__container">
        {/* Header */}
        <div className="analytics__header">
          <div className="analytics__badge">
            <HiSparkles /> AI Fraud Intelligence Center
          </div>
          <h2 className="analytics__title">Fraud Prediction Dashboard</h2>
          <p className="analytics__subtitle">
            Comprehensive real-time analytics monitoring claim distributions, seasonal fraud trends, accuracy metrics, and AI feature weights.
          </p>
        </div>

        {/* ── ROW 1: 4 Top Stat Cards with Sparklines (Image 4 Top Row) ── */}
        <div className="dash__stats-grid">
          {/* Card 1: Total Claims */}
          <div className="dash__stat-card glass">
            <div className="dash__stat-top">
              <div className="dash__stat-icon dash__stat-icon--blue">
                <FaShieldAlt />
              </div>
              <div className="dash__stat-title">Claims Overview</div>
            </div>
            <div className="dash__stat-num">11,716</div>
            <div className="dash__stat-sub">Total Claims Evaluated</div>
            <div className="dash__sparkline">
              <ResponsiveContainer width="100%" height={36}>
                <AreaChart data={totalClaimsSparkline}>
                  <defs>
                    <linearGradient id="blueSpark" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#6366f1" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="v" stroke="#6366f1" strokeWidth={2} fill="url(#blueSpark)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Card 2: Fraudulent Claims */}
          <div className="dash__stat-card glass">
            <div className="dash__stat-top">
              <div className="dash__stat-icon dash__stat-icon--red">
                <FaExclamationTriangle />
              </div>
              <div className="dash__stat-title">Fraudulent Claims</div>
            </div>
            <div className="dash__stat-num">2,878</div>
            <div className="dash__stat-sub">Fraud Detected (24.5%)</div>
            <div className="dash__sparkline">
              <ResponsiveContainer width="100%" height={36}>
                <AreaChart data={fraudClaimsSparkline}>
                  <defs>
                    <linearGradient id="redSpark" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="v" stroke="#ef4444" strokeWidth={2} fill="url(#redSpark)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Card 3: Detection Accuracy */}
          <div className="dash__stat-card glass">
            <div className="dash__stat-top">
              <div className="dash__stat-icon dash__stat-icon--green">
                <FaCheckCircle />
              </div>
              <div className="dash__stat-title">Detection Accuracy</div>
            </div>
            <div className="dash__stat-num">78.1%</div>
            <div className="dash__stat-sub">Model Validation Accuracy</div>
            <div className="dash__sparkline">
              <ResponsiveContainer width="100%" height={36}>
                <AreaChart data={accuracySparkline}>
                  <defs>
                    <linearGradient id="greenSpark" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                      <stop offset="100%" stopColor="#10b981" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="v" stroke="#10b981" strokeWidth={2} fill="url(#greenSpark)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Card 4: Vehicle Category Pie Chart (Image 4 Right Card) */}
          <div className="dash__stat-card glass">
            <div className="dash__stat-top">
              <div className="dash__stat-icon dash__stat-icon--purple">
                <FaCar />
              </div>
              <div className="dash__stat-title">Fraud by Vehicle Category</div>
            </div>
            <div className="dash__pie-container">
              <ResponsiveContainer width="45%" height={80}>
                <PieChart>
                  <Pie data={vehicleCategoryData} innerRadius={18} outerRadius={36} dataKey="value">
                    {vehicleCategoryData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div className="dash__pie-legend">
                {vehicleCategoryData.map((item, idx) => (
                  <div key={idx} className="dash__legend-row">
                    <span className="dash__dot" style={{ background: item.color }}></span>
                    <span className="dash__legend-name">{item.name}</span>
                    <span className="dash__legend-val">{item.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── ROW 2: Mac Window Dashboard + Right Side Donut/Accuracy (Image 2 & 3) ── */}
        <div className="dash__main-layout">
          {/* Left: Mac Window Fraud Prediction Dashboard */}
          <div className="dash__mac-window">
            {/* Mac Window Header */}
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
              <ResponsiveContainer width="100%" height={320}>
                <AreaChart data={dashboardTrendData} margin={{ top: 25, right: 25, left: -15, bottom: 5 }}>
                  <defs>
                    <linearGradient id="purpleSpline" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#818cf8" stopOpacity={0.6} />
                      <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
                  <XAxis dataKey="month" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12, fontWeight: 600 }} />
                  <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 550]} />
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

          {/* Right Side: Donut Distribution & Circular Gauge (Image 3 Right Column) */}
          <div className="dash__right-col">
            {/* Top Right: Claim Distribution Donut */}
            <div className="dash__side-card glass">
              <h3 className="dash__side-title">Claim Distribution</h3>
              <div className="dash__donut-box">
                <ResponsiveContainer width={120} height={120}>
                  <PieChart>
                    <Pie data={claimDistributionData} innerRadius={35} outerRadius={52} dataKey="value">
                      {claimDistributionData.map((entry, idx) => (
                        <Cell key={`cell-${idx}`} fill={entry.color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="dash__side-legend">
                  <div className="dash__side-item">
                    <span className="dash__dot" style={{ background: '#6366f1' }}></span>
                    <div>
                      <div className="dash__side-name">Not Fraud</div>
                      <div className="dash__side-num">72.5%</div>
                    </div>
                  </div>
                  <div className="dash__side-item">
                    <span className="dash__dot" style={{ background: '#ef4444' }}></span>
                    <div>
                      <div className="dash__side-name">Fraud</div>
                      <div className="dash__side-num">27.5%</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Bottom Right: Model Accuracy Radial Meter */}
            <div className="dash__side-card glass">
              <h3 className="dash__side-title">Model Accuracy</h3>
              <div className="dash__radial-box">
                <div className="dash__radial-ring">
                  <span className="dash__radial-val">78.1%</span>
                </div>
                <div className="dash__radial-label">High Accuracy Decision Tree</div>
              </div>
            </div>
          </div>
        </div>

        {/* ── ROW 3: Fraud vs Not Fraud Over Time Line Chart (Image 1) ── */}
        <div className="dash__line-card glass">
          <div className="dash__line-header">
            <div>
              <h3 className="dash__line-title">Fraud vs Not Fraud Over Time</h3>
              <p className="dash__line-sub">Monthly trend comparison of legitimate insurance claims vs detected fraud claims.</p>
            </div>
            <div className="dash__line-legend-top">
              <span className="dash__legend-badge dash__legend-badge--blue">● Not Fraud</span>
              <span className="dash__legend-badge dash__legend-badge--red">● Fraud</span>
            </div>
          </div>

          <div className="dash__line-chart-wrapper">
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={monthlyComparisonData} margin={{ top: 15, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.08)" vertical={false} />
                <XAxis dataKey="month" stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12, fontWeight: 600 }} />
                <YAxis stroke="#94a3b8" tick={{ fill: '#94a3b8', fontSize: 12 }} domain={[0, 800]} />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="NotFraud"
                  name="Not Fraud"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#3b82f6', strokeWidth: 2, stroke: '#ffffff' }}
                  activeDot={{ r: 8 }}
                />
                <Line
                  type="monotone"
                  dataKey="Fraud"
                  name="Fraud"
                  stroke="#ef4444"
                  strokeWidth={3}
                  dot={{ r: 5, fill: '#ef4444', strokeWidth: 2, stroke: '#ffffff' }}
                  activeDot={{ r: 8 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ModelAnalytics;
