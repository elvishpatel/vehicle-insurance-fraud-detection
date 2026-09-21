import React from 'react';
import { FaArrowRight } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './Hero.css';

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
          AI-powered fraud prediction system that analyzes insurance claim details using Machine Learning and LightGBM Gradient Boosting.
        </p>
        
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
            <div className="hero__stat-value gradient-text">90.8%</div>
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
