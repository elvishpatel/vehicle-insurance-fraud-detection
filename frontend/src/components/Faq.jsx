import React, { useState } from 'react';
import { FaChevronDown, FaQuestionCircle } from 'react-icons/fa';
import { HiSparkles } from 'react-icons/hi';
import './Faq.css';

const faqs = [
  {
    question: 'How does the fraud detection actually work?',
    answer:
      'Our engine was trained on 15,420 historical vehicle insurance claims, including 923 confirmed fraud cases. It learned the statistical patterns that separate fraudulent claims from legitimate ones — such as fault assignment, policy type, claim timing, and address-change history. When you submit a claim, it compares those same patterns against everything it learned and returns a fraud probability.',
  },
  {
    question: 'What do the risk levels mean?',
    answer:
      'Every claim is scored between 0% and 100% fraud probability. Scores above the decision threshold are flagged as Fraud (High risk). Scores approaching the threshold are marked Medium risk and worth a quick manual review, while everything else is Low risk. The exact probability is always shown so you can apply your own judgment.',
  },
  {
    question: 'Why does the form only ask for 15 details?',
    answer:
      'We measured which inputs actually influence the prediction. The 15 fields in the form carry essentially all of the predictive signal; the remaining background factors changed results by less than half a percent, so they are auto-filled with typical values. You get the same verdict with a fraction of the typing.',
  },
  {
    question: 'Is my claim data stored anywhere?',
    answer:
      'No. Claims are scored in memory and the result is returned to you immediately — nothing is written to a database, and no personal data is retained after the analysis completes.',
  },
  {
    question: 'Why is the first prediction sometimes slow?',
    answer:
      'The service runs on free hosting, which sleeps after a period of inactivity. The very first request after a break may take up to a minute while the server wakes up — the app will show you a "server waking up" message and complete your request automatically, so you never have to resubmit.',
  },
];

const Faq = () => {
  const [openIndex, setOpenIndex] = useState(0);

  const toggle = (index) => {
    setOpenIndex((prev) => (prev === index ? -1 : index));
  };

  return (
    <section id="faq" className="faq">
      <div className="faq__header">
        <div className="faq__badge">
          <HiSparkles /> Good to Know
        </div>
        <h2 className="faq__title">Frequently Asked Questions</h2>
        <p className="faq__subtitle">
          Everything you need to understand about how FraudShield evaluates your claims
        </p>
      </div>

      <div className="faq__list">
        {faqs.map((faq, index) => {
          const isOpen = openIndex === index;
          return (
            <div
              key={index}
              className={`faq__item ${isOpen ? 'faq__item--open' : ''}`}
            >
              <button
                type="button"
                className="faq__question"
                onClick={() => toggle(index)}
                aria-expanded={isOpen}
              >
                <span className="faq__question-icon">
                  <FaQuestionCircle />
                </span>
                <span className="faq__question-text">{faq.question}</span>
                <FaChevronDown className="faq__chevron" />
              </button>
              <div className="faq__answer-wrapper">
                <div className="faq__answer">
                  <p>{faq.answer}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
};

export default Faq;
