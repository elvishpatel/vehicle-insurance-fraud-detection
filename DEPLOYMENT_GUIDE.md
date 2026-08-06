# 🚀 Deployment Guide: Insurance Fraud Detection System

Follow this step-by-step guide to deploy your Flask Backend (Machine Learning model) and React Frontend live to the internet for **FREE**.

---

## 🏗️ Architecture Overview

```
┌────────────────────────────────┐         ┌────────────────────────────────┐
│   React Frontend (Vercel)      │  HTTP   │    Flask Backend (Render)      │
│   https://fraudshield.vercel.app│ ──────> │https://fraudshield-api.onrender│
│   (Vite + React 18)            │  POST   │ (DecisionTree ML Model + CORS) │
└────────────────────────────────┘         └────────────────────────────────┘
```

---

## 📌 STEP 1: Push Code to GitHub

1. Create a new repository on [GitHub](https://github.com/new) named `vehicle-insurance-fraud-detection`.
2. Open terminal in project root (`e:\Programmes\ML\Project\Vehicle Insurance Fraud Detection`):

```bash
git init
git add .
git commit -m "Initial commit - Complete Fraud Detection SaaS"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vehicle-insurance-fraud-detection.git
git push -u origin main
```

---

## 🐍 STEP 2: Deploy Flask Backend to Render (Free)

[Render](https://render.com) provides free hosting for Python Flask web applications.

1. Sign up / Log in to [Render.com](https://render.com).
2. Click **New +** → Select **Web Service**.
3. Connect your GitHub repository `vehicle-insurance-fraud-detection`.
4. Configure the settings:
   - **Name**: `insurance-fraud-backend` (or your choice)
   - **Region**: Select closest to your users
   - **Branch**: `main`
   - **Root Directory**: Leave blank (root directory)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`
5. Click **Create Web Service**.
6. Wait 2-3 minutes for the build to finish.
7. Copy your live backend URL (e.g. `https://insurance-fraud-backend.onrender.com`).

> **Verify**: Open `https://insurance-fraud-backend.onrender.com/health` in your browser. You should get:
> `{"features":50,"model":"DecisionTreeClassifier","status":"healthy"}`

---

## ⚛️ STEP 3: Deploy React Frontend to Vercel (Free)

[Vercel](https://vercel.com) provides free hosting for Vite React applications.

1. Sign up / Log in to [Vercel.com](https://vercel.com).
2. Click **Add New...** → **Project**.
3. Import your GitHub repository `vehicle-insurance-fraud-detection`.
4. Configure Project Settings:
   - **Framework Preset**: `Vite`
   - **Root Directory**: Click Edit → Select `frontend` folder
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. Expand **Environment Variables**:
   - **Key**: `VITE_API_BASE_URL`
   - **Value**: `https://insurance-fraud-backend.onrender.com` (Your Render backend URL from Step 2)
6. Click **Deploy**.
7. In ~1 minute, Vercel will give you your live URL (e.g. `https://vehicle-insurance-fraud-detection.vercel.app`).

---

## 🧪 STEP 4: Test Your Live App

1. Open your Vercel URL in your browser.
2. Navigate to the **Predict Insurance Fraud** form section.
3. Fill out the fields and click **Analyze Claim**.
4. You will see the animated result card showing **Fraud Detected (High Risk)** or **Claim is Legitimate (Low Risk)** powered by your live Flask API!

---

## 🛠️ Troubleshooting & Notes

- **Render Cold Starts**: Render's free tier puts web services to sleep after 15 minutes of inactivity. The first prediction after inactivity may take ~20-30 seconds while the server spins up. The Axios timeout has been increased to 15s to handle this gracefully.
- **Updating Code**: Whenever you push changes to your GitHub `main` branch, both Render and Vercel will automatically re-deploy your latest code!
