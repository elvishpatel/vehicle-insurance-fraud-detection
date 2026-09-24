import React from 'react';
import { FaArrowRight, FaCheckCircle } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './Hero.css';

const highlights = [
  'Real-time claim scoring',
  'Trained on 15,420 real claims',
  'Clear risk-band verdicts',
];

const Hero = () => {
  const scrollToPrediction = () => {
    document.getElementById('prediction')?.scrollIntoView({ behavior: 'smooth' });
  };

  const scrollToFeatures = () => {
    document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section id="hero" className="hero">
      <div className="hero__bg">
        <div className="hero__blob hero__blob--1"></div>
        <div className="hero__blob hero__blob--2"></div>
        <div className="hero__blob hero__blob--3"></div>
      </div>

      <div className="hero__content">
        <div className="hero__badge">
          <HiSparkles />
          <span>AI-Powered Fraud Detection</span>
        </div>

        <h1 className="hero__title">
          Insurance Fraud <br />
          <span className="gradient-text">Detection System</span>
        </h1>

        <p className="hero__subtitle">
          Advanced machine learning that screens every vehicle insurance claim in
          seconds — flagging suspicious patterns before they cost your business money.
        </p>

        <ul className="hero__highlights">
          {highlights.map((item) => (
            <li key={item} className="hero__highlight">
              <FaCheckCircle /> {item}
            </li>
          ))}
        </ul>

        <div className="hero__buttons">
          <button className="hero__btn-primary" onClick={scrollToPrediction}>
            Predict Claim <FaArrowRight />
          </button>
          <button className="hero__btn-outline" onClick={scrollToFeatures}>
            Learn More
          </button>
        </div>

        <div className="hero__stats">
          <div className="hero__stat-card hero__stat-card--1">
            <div className="hero__stat-value gradient-text">82.5%</div>
            <div className="hero__stat-label">Accuracy</div>
          </div>
          <div className="hero__stat-card hero__stat-card--2">
            <div className="hero__stat-value gradient-text">&lt; 2s</div>
            <div className="hero__stat-label">Analysis</div>
          </div>
          <div className="hero__stat-card hero__stat-card--3">
            <div className="hero__stat-value gradient-text">15.4K+</div>
            <div className="hero__stat-label">Claims Trained</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
