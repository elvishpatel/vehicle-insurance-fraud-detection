import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ModelAnalytics from './components/ModelAnalytics';
import Features from './components/Features';
import Timeline from './components/Timeline';
import PredictionForm from './components/PredictionForm';
import ResultCard from './components/ResultCard';
import Footer from './components/Footer';
import { FaArrowUp } from 'react-icons/fa';
import './App.css';

/**
 * App — Root component
 * Manages dark/light theme, prediction result state, and back-to-top button.
 */
function App() {
  // ── Theme ──────────────────────────────────────────────
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('fraudshield-theme');
    return saved ? saved === 'dark' : true; // default dark
  });

  useEffect(() => {
    document.documentElement.setAttribute(
      'data-theme',
      darkMode ? 'dark' : 'light'
    );
    localStorage.setItem('fraudshield-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  const toggleDarkMode = () => setDarkMode((prev) => !prev);

  // ── Prediction result ─────────────────────────────────
  const [result, setResult] = useState(null);

  const handleResult = (data) => {
    setResult(data);
  };

  // ── Back to top button ────────────────────────────────
  const [showBackToTop, setShowBackToTop] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setShowBackToTop(window.scrollY > 400);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // ── Render ────────────────────────────────────────────
  return (
    <div className="app">
      <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />
      <Hero />
      <ModelAnalytics />
      <Features />
      <Timeline />
      <PredictionForm onResult={handleResult} />
      <ResultCard result={result} />
      <Footer />

      {/* Back to top button */}
      <button
        className={`back-to-top ${showBackToTop ? 'back-to-top--visible' : ''}`}
        onClick={scrollToTop}
        aria-label="Back to top"
      >
        <FaArrowUp />
      </button>
    </div>
  );
}

export default App;
