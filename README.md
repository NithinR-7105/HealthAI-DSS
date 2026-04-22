# HealthAI DSS 🩺

**An Explainable AI-Based Clinical Decision Support System for Early 
Diabetes Risk Detection Using XGBoost, SHAP, and LLM-Generated 
Personalised Recommendations**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)
![Gemini](https://img.shields.io/badge/Google%20Gemini-2.5%20Flash-purple)

---

## 📌 Overview

HealthAI DSS is a complete, deployable web-based Clinical Decision 
Support System that detects early diabetes risk with high sensitivity, 
explains every prediction at the patient level using SHAP, and 
generates personalised clinical recommendations using Google Gemini 
2.5 Flash.

This is the first reported system on the Pima Indians Diabetes Dataset 
that integrates all three components — sensitivity-first prediction, 
patient-level explainability, and an LLM recommendation layer — into 
a single deployable clinical interface.

---

## ✨ Key Features

- 🎯 **Sensitivity-First Prediction** — XGBoost + SMOTETomek achieves 
  sensitivity of 0.8704, exceeding the clinical screening threshold of 0.80
- 🔍 **Patient-Level Explainability** — SHAP TreeExplainer generates 
  per-patient waterfall plots identifying top contributing risk factors
- 🤖 **LLM Recommendations** — Google Gemini 2.5 Flash generates 
  personalised tier-specific clinical guidance for every patient
- 💬 **AI Chatbot** — Context-aware chatbot answers patient follow-up 
  questions using their exact clinical values
- 🌐 **Web Interface** — Clean Flask-based single-page clinical risk 
  report with risk score, tier badge, SHAP chart, and recommendations
- 📊 **Three-Tier Risk Stratification** — Low (< 0.40), 
  Moderate (0.40–0.70), High (> 0.70)

---

## 🏗️ System Architecture

Patient Data Input (Flask)
↓
Data Preprocessing & Feature Engineering
(Zero Replacement → Median Imputation →
StandardScaler → Glucose_BMI + Age_Pedigree)
↓
Prediction Engine
(SMOTETomek → XGBoost → Threshold t*=0.55 → Risk Tier)
↓
Explainability Layer
(SHAP TreeExplainer → Shapley Values → Waterfall Plot)
↓
Recommendation Layer
(Google Gemini 2.5 Flash → Personalised Clinical Guidance)
↓
Flask Risk Report Output
(Risk Score + SHAP Chart + Recommendations + AI Chatbot)

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Sensitivity (TPR) | **0.8704** |
| Balanced Accuracy | **0.7952** |
| ROC-AUC | **0.8300** |
| Specificity (TNR) | 0.7200 |
| Accuracy | 0.7727 |
| F1-Score | 0.7287 |
| Optimal Threshold | t* = 0.55 |
| False Negatives | 7 only (out of 54) |

> Evaluated on held-out test set of 154 samples from the 
> Pima Indians Diabetes Dataset.

---

## 🔬 Dataset

- **Name:** Pima Indians Diabetes Dataset (PIDD)
- **Samples:** 768 (500 healthy, 268 diabetic)
- **Features:** 8 clinical features + 2 engineered interaction features
- **Class Ratio:** 65:35 (addressed using SMOTETomek)
- **Source:** UCI Machine Learning Repository

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Web Framework | Flask (Python) |
| ML Model | XGBoost |
| Class Balancing | SMOTETomek (imbalanced-learn) |
| Explainability | SHAP TreeExplainer |
| LLM | Google Gemini 2.5 Flash |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualisation | Matplotlib |
| PDF Reports | ReportLab |
| Model Storage | Joblib |

---

## 📁 Project Structure

HealthAI-DSS/
│
├── app.py                        # Main Flask application
├── stage2_smote.py               # SMOTETomek training pipeline
├── stage3_shap.py                # SHAP explainability analysis
├── stage4_llm_recommendations.py # Gemini LLM + PDF report
├── diabetes.csv                  # Pima Indians Diabetes Dataset
│
├── templates/
│   ├── ui.html                   # Patient data input form
│   └── risk.html                 # Risk report + AI chatbot
│
├── static/
│   └── shap/                     # Generated SHAP waterfall plots
│
├── model/                        # Saved model artefacts (joblib)
├── requirements.txt              # Python dependencies
└── README.md

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/NithinR-7105/HealthAI-DSS.git
cd HealthAI-DSS
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Gemini API key
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Mac / Linux
export GEMINI_API_KEY=your_api_key_here
```

### 4. Run the application
```bash
python app.py
```

### 5. Open in browser
http://localhost:5000


---

## 📦 Requirements

flask
pandas
numpy
xgboost
scikit-learn
imbalanced-learn
shap
matplotlib
google-generativeai
joblib
reportlab

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📈 SHAP Feature Importance

| Rank | Feature | Mean |SHAP Value| |
|------|---------|-------------------|
| 1 | Glucose_BMI (engineered) | 0.8636 |
| 2 | Age | 0.2580 |
| 3 | Glucose | 0.2317 |
| 4 | Insulin | ~0.19 |
| 5 | Age_Pedigree (engineered) | ~0.18 |

> Glucose_BMI = Glucose × BMI captures non-linear 
> compounded metabolic risk unavailable from individual features.

---

## 👨‍💻 Author:

Nithin R - Final Year
Department of Artificial Intelligence and Data Science

---

## ⚠️ Disclaimer

This system is a research prototype intended to support 
clinical decision-making. It does not replace professional 
medical diagnosis. Always consult a qualified healthcare 
professional for clinical evaluation and treatment.


---

⭐ If you found this project useful, please give it a star!

