import React, { useEffect, useRef } from 'react';
import { FaClipboardList, FaPaperPlane, FaCog, FaCheckCircle } from 'react-icons/fa';
import './Timeline.css';

const steps = [
  {
    icon: FaClipboardList,
    title: 'Enter Claim Details',
    description: 'Fill in the insurance claim information including driver details, vehicle data, and claim specifics.',
  },
  {
    icon: FaPaperPlane,
    title: 'Submit to API',
    description: 'The frontend securely sends the data to our Flask API endpoint for processing and analysis.',
  },
  {
    icon: FaCog,
    title: 'ML Model Predicts',
    description: 'Our trained machine learning engine analyzes the claim patterns and calculates a fraud probability in real time.',
  },
  {
    icon: FaCheckCircle,
    title: 'View Results',
    description: 'Instantly receive a clear Fraud or Not Fraud prediction with confidence score and risk assessment.',
  }
];

const Timeline = () => {
  const containerRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('timeline__step--visible');
          }
        });
      },
      { threshold: 0.2 }
    );

    const stepElements = containerRef.current.querySelectorAll('.timeline__step');
    stepElements.forEach((el) => observer.observe(el));

    return () => {
      stepElements.forEach((el) => observer.unobserve(el));
    };
  }, []);

  return (
    <section id="how-it-works" className="timeline">
      <div className="timeline__header">
        <h2>How It Works</h2>
        <p>Simple four-step process to detect fraudulent insurance claims</p>
      </div>
      <div className="timeline__container" ref={containerRef}>
        <div className="timeline__line"></div>
        {steps.map((step, index) => {
          const isRight = index % 2 === 1;
          return (
            <div 
              key={index} 
              className={`timeline__step ${isRight ? 'timeline__step--right' : 'timeline__step--left'}`}
            >
              <div className="timeline__content">
                <div className="timeline__icon">
                  <step.icon />
                </div>
                <h3 className="timeline__step-title">{step.title}</h3>
                <p className="timeline__step-description">{step.description}</p>
              </div>
              <div className="timeline__node">
                <span className="timeline__number">{index + 1}</span>
              </div>
              <div className="timeline__spacer"></div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default Timeline;
