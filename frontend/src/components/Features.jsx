import React, { useEffect, useRef } from 'react';
import { FaBrain, FaLayerGroup, FaBolt, FaShieldAlt, FaDatabase, FaUserSecret } from 'react-icons/fa';
import './Features.css';

const features = [
  { 
    icon: FaBrain, 
    title: 'AI-Powered Prediction', 
    description: 'Advanced machine learning algorithms analyze claim patterns to identify potential fraud with high accuracy and confidence scores.' 
  },
  { 
    icon: FaLayerGroup, 
    title: 'Clear Risk Bands', 
    description: 'Every claim receives a High, Medium, or Low risk verdict with an exact fraud probability — so investigators know where to focus first.' 
  },
  { 
    icon: FaBolt, 
    title: 'Fast Real-Time Analysis', 
    description: 'Get instant predictions in under 2 seconds. Our optimized pipeline processes claims data rapidly for immediate results.' 
  },
  { 
    icon: FaDatabase, 
    title: 'Trained on Real Claims', 
    description: 'Over 90% accuracy powered by training on 15,400+ real insurance claims with rigorous out-of-fold validation.' 
  },
  { 
    icon: FaUserSecret, 
    title: 'Pattern Recognition', 
    description: 'Detects subtle fraud signals humans miss — from fault and policy details to claim timing and address history.' 
  },
  { 
    icon: FaShieldAlt, 
    title: 'Privacy-First Design', 
    description: 'Claims are scored in memory and never stored. No personal data is retained after your analysis is complete.' 
  }
];

const Features = () => {
  const cardsRef = useRef([]);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('features__card--visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    cardsRef.current.forEach(card => {
      if (card) {
        observer.observe(card);
      }
    });

    return () => {
      if (cardsRef.current) {
        cardsRef.current.forEach(card => {
          if (card) observer.unobserve(card);
        });
      }
    };
  }, []);

  return (
    <section id="features" className="features">
      <div className="features__header">
        <h2 className="features__title">Powerful Features</h2>
        <p className="features__subtitle">Built with cutting-edge technology to deliver accurate and reliable fraud detection</p>
      </div>
      <div className="features__grid">
        {features.map((feature, index) => {
          const Icon = feature.icon;
          return (
            <div 
              key={index} 
              className="features__card"
              ref={el => cardsRef.current[index] = el}
              style={{ transitionDelay: `${index * 0.1}s` }}
            >
              <div className="features__card-icon">
                <Icon />
              </div>
              <h3 className="features__card-title">{feature.title}</h3>
              <p className="features__card-description">{feature.description}</p>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default Features;
