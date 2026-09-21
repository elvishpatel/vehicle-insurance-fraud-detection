import React, { useState, useEffect } from 'react';
import { HiMenuAlt3, HiX } from 'react-icons/hi';
import { FiSun, FiMoon } from 'react-icons/fi';
import { SiShieldsdotio } from 'react-icons/si';
import './Navbar.css';

const Navbar = ({ darkMode, toggleDarkMode }) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768 && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [mobileMenuOpen]);

  const scrollToSection = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
    setMobileMenuOpen(false);
  };

  const navLinks = [
    { name: 'Features', id: 'features' },
    { name: 'How It Works', id: 'how-it-works' },
    { name: 'FAQ', id: 'faq' },
  ];

  return (
    <>
      <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
        <div className="navbar__container">
          <div className="navbar__logo">
            <SiShieldsdotio className="navbar__logo-icon" />
            <span className="gradient-text">FraudShield</span>
          </div>

          <ul className="navbar__links">
            {navLinks.map((link) => (
              <li key={link.name}>
                <button
                  className="navbar__link"
                  onClick={() => scrollToSection(link.id)}
                >
                  {link.name}
                </button>
              </li>
            ))}
          </ul>

          <div className="navbar__actions">
            <button
              className="navbar__theme-toggle"
              onClick={toggleDarkMode}
              aria-label="Toggle Dark Mode"
            >
              {darkMode ? <FiSun /> : <FiMoon />}
            </button>
            <button
              className="navbar__mobile-btn"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle Mobile Menu"
            >
              {mobileMenuOpen ? <HiX /> : <HiMenuAlt3 />}
            </button>
          </div>
        </div>
      </nav>

      {/* Mobile Menu Overlay */}
      {mobileMenuOpen && (
        <div
          className="navbar__mobile-overlay"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Mobile Menu */}
      <div className={`navbar__mobile-menu ${mobileMenuOpen ? 'navbar__mobile-menu--open' : ''}`}>
        <div className="navbar__mobile-header">
          <div className="navbar__logo">
            <SiShieldsdotio className="navbar__logo-icon" />
            <span className="gradient-text">FraudShield</span>
          </div>
          <button
            className="navbar__mobile-btn"
            onClick={() => setMobileMenuOpen(false)}
            aria-label="Close Mobile Menu"
            style={{ display: 'flex' }}
          >
            <HiX />
          </button>
        </div>
        <ul className="navbar__mobile-links">
          {navLinks.map((link) => (
            <li key={link.name}>
              <button
                className="navbar__mobile-link"
                onClick={() => scrollToSection(link.id)}
              >
                {link.name}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};

export default Navbar;
