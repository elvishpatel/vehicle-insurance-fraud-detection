import React from 'react';
import { FaGithub, FaLinkedin, FaHeart, FaShieldAlt } from 'react-icons/fa';
import { SiFlask, SiReact, SiPython, SiScikitlearn } from 'react-icons/si';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer__container">
        <div className="footer__grid">
          {/* Column 1 */}
          <div className="footer__brand">
            <div className="footer__logo">
              <FaShieldAlt className="footer__logo-icon" />
              <span>FraudShield</span>
            </div>
            <p className="footer__description">
              Advanced vehicle insurance fraud detection powered by Machine Learning. 
              Protecting businesses and ensuring fair premiums for everyone.
            </p>
          </div>

          {/* Column 2 */}
          <div className="footer__column">
            <h4 className="footer__column-title">Tech Stack</h4>
            <ul className="footer__list">
              <li className="footer__list-item">
                <span className="footer__list-link"><SiScikitlearn /> Decision Tree</span>
              </li>
              <li className="footer__list-item">
                <span className="footer__list-link"><SiFlask /> Flask API</span>
              </li>
              <li className="footer__list-item">
                <span className="footer__list-link"><SiReact /> React Frontend</span>
              </li>
              <li className="footer__list-item">
                <span className="footer__list-link"><SiPython /> Python</span>
              </li>
            </ul>
          </div>

          {/* Column 3 */}
          <div className="footer__column">
            <h4 className="footer__column-title">Quick Links</h4>
            <ul className="footer__list">
              <li className="footer__list-item">
                <a href="#features" className="footer__list-link">Features</a>
              </li>
              <li className="footer__list-item">
                <a href="#how-it-works" className="footer__list-link">How It Works</a>
              </li>
              <li className="footer__list-item">
                <a href="#predict" className="footer__list-link">Predict Now</a>
              </li>
            </ul>
          </div>

          {/* Column 4 */}
          <div className="footer__column">
            <h4 className="footer__column-title">Connect</h4>
            <div className="footer__socials">
              <a 
                href="https://github.com/elvishpatel/vehicle-insurance-fraud-detection" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="footer__social-link"
                aria-label="GitHub Repository"
              >
                <FaGithub />
              </a>
              <a 
                href="https://www.linkedin.com/in/elvish-kansagara" 
                target="_blank" 
                rel="noopener noreferrer" 
                className="footer__social-link"
                aria-label="LinkedIn Profile"
              >
                <FaLinkedin />
              </a>
            </div>
          </div>
        </div>

        <div className="footer__divider"></div>

        <div className="footer__bottom">
          <div className="footer__made-with">
            Made with <FaHeart className="footer__heart" /> by{' '}
            <a 
              href="https://elvishpatel.in" 
              target="_blank" 
              rel="noopener noreferrer"
              className="footer__author-link"
            >
              <span className="footer__author-first">Elvish</span>{' '}
              <span className="footer__author-last">Patel</span>
            </a>
          </div>
          <div className="footer__copyright">
            &copy; 2026 All rights reserved
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
