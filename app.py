import os
import re
import json
import time
import joblib
import warnings
import threading
import numpy as np
import pandas as pd
import shap
import uuid

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import google.generativeai as genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    GEMINI_API_KEY = "AIzaSyB0uEqF9AGBaqp-IhR2T3yagGxSyw7TOSs"

DATA_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
MODEL_DIR     = "model"
SHAP_PLOT_DIR = os.path.join("static", "shap")

os.makedirs(MODEL_DIR,     exist_ok=True)
os.makedirs(SHAP_PLOT_DIR, exist_ok=True)

if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY is not set. Gemini features will be unavailable.")
    print("[WARNING] Set it with: set GEMINI_API_KEY=your_key  (Windows)")
    print("[WARNING]              export GEMINI_API_KEY=your_key  (Mac/Linux)")
else:
    print(f"[INFO] Gemini API key loaded (ends with ...{GEMINI_API_KEY[-4:]})")
genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL   = "gemini-2.5-flash"

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", os.urandom(24).hex())

THRESHOLD = 0.55 

PLOT_LOCK = threading.Lock()

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    "Glucose_BMI", "Age_Pedigree"
]

INVALID_ZERO_FIELDS = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

CLINICAL_UNITS = {
    "Pregnancies":              "count",
    "Glucose":                  "mg/dL",
    "BloodPressure":            "mmHg",
    "SkinThickness":            "mm",
    "Insulin":                  "mu U/mL",
    "BMI":                      "kg/m2",
    "DiabetesPedigreeFunction": "score",
    "Age":                      "years",
    "Glucose_BMI":              "composite",
    "Age_Pedigree":             "composite",
}



def get_clinical_status(feature, value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not provided"
    if feature == "Glucose":
        if value >= 126:    return "Diabetic range"
        elif value >= 100:  return "Pre-diabetic"
        elif value >= 70:   return "Normal"
        else:               return "Low"
    elif feature == "BMI":
        if value >= 30:     return "Obese"
        elif value >= 25:   return "Overweight"
        elif value >= 18.5: return "Normal weight"
        else:               return "Underweight"
    elif feature == "BloodPressure":
        if value >= 90:     return "Hypertensive"
        elif value >= 60:   return "Normal"
        else:               return "Low"
    elif feature == "DiabetesPedigreeFunction":
        if value > 0.8:     return "Strong family history"
        elif value > 0.4:   return "Moderate family history"
        else:               return "Low family history"
    elif feature == "Age":
        if value >= 60:     return "Senior"
        elif value >= 45:   return "Middle-aged"
        else:               return "Younger adult"
    elif feature == "Insulin":
        if value > 166:     return "Above normal"
        elif value >= 16:   return "Normal"
        else:               return "Below normal"
    elif feature == "Glucose_BMI":
        if value >= 5000:   return "Both elevated — high combined risk"
        elif value >= 3000: return "Moderately elevated"
        else:               return "Within acceptable range"
    elif feature == "Age_Pedigree":
        if value >= 40:     return "Strong age-family risk interaction"
        elif value >= 20:   return "Moderate age-family interaction"
        else:               return "Low age-family interaction"
    return "-"


def get_risk_tier(prob):
    if prob >= 0.70:   return "HIGH RISK",     "high"
    elif prob >= 0.40: return "MODERATE RISK", "moderate"
    else:              return "LOW RISK",      "low"


def get_confidence_label(prob):
    if prob >= 0.80:   return "High Confidence"
    elif prob >= 0.60: return "Moderate Confidence"
    elif prob >= 0.50: return "Low Confidence"
    else:              return "Likely Non-Diabetic"



def load_or_train_model():
    paths = {
        "model":   os.path.join(MODEL_DIR, "model.pkl"),
        "scaler":  os.path.join(MODEL_DIR, "scaler.pkl"),
        "imputer": os.path.join(MODEL_DIR, "imputer.pkl"),
    }

    if all(os.path.exists(p) for p in paths.values()):
        print("[INFO] Loading saved model from model/ ...")
        model   = joblib.load(paths["model"])
        scaler  = joblib.load(paths["scaler"])
        imputer = joblib.load(paths["imputer"])
        print("[INFO] Model, scaler, imputer loaded.")
    else:
        print("[WARN] pkl files missing — retraining from diabetes.csv ...")
        df = pd.read_csv(DATA_PATH)

        df[list(INVALID_ZERO_FIELDS)] = df[list(INVALID_ZERO_FIELDS)].replace(0, np.nan)

        df["Glucose_BMI"]  = df["Glucose"].fillna(df["Glucose"].median()) *                              df["BMI"].fillna(df["BMI"].median())
        df["Age_Pedigree"] = df["Age"] * df["DiabetesPedigreeFunction"]

        X = df.drop("Outcome", axis=1)  
        y = df["Outcome"]
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        X_imp   = imputer.fit_transform(X_train)
        X_sc    = scaler.fit_transform(X_imp)
        X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_sc, y_train)

        model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            min_child_weight=2, gamma=0.0,
            subsample=0.8, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, verbosity=0
        )
        model.fit(X_res, y_res)

        joblib.dump(model,   paths["model"])
        joblib.dump(scaler,  paths["scaler"])
        joblib.dump(imputer, paths["imputer"])
        print("[INFO] Model retrained and saved to model/")

    return model, scaler, imputer



MODEL, SCALER, IMPUTER = load_or_train_model()
EXPLAINER = shap.TreeExplainer(MODEL)
print("[INFO] SHAP TreeExplainer ready.")




def preprocess_input(raw_input: dict):
    """
    Preprocessing steps (must match stage2_smote.py order exactly):
      Step 1: Extract 8 raw values, replace invalid zeros with NaN
      Step 2: Feature engineering FIRST (Glucose_BMI, Age_Pedigree) — matches stage2
      Step 3: Build 10-feature array
      Step 4: Impute (fitted on 10 features)
      Step 5: Scale (fitted on 10 features)
    Returns:
        scaled   (1,10) — model input
        original (1,10) — imputed but unscaled, for display
    """
    BASE_FEATURES = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    vals = {}
    for f in BASE_FEATURES:
        v = float(raw_input[f])
        vals[f] = np.nan if (f in INVALID_ZERO_FIELDS and v == 0.0) else v

    g = vals["Glucose"] if not np.isnan(vals["Glucose"]) else 0.0
    b = vals["BMI"]     if not np.isnan(vals["BMI"])     else 0.0
    glucose_bmi  = g * b
    age_pedigree = vals["Age"] * vals["DiabetesPedigreeFunction"]

    row = np.array([[
        vals["Pregnancies"], vals["Glucose"], vals["BloodPressure"],
        vals["SkinThickness"], vals["Insulin"], vals["BMI"],
        vals["DiabetesPedigreeFunction"], vals["Age"],
        glucose_bmi, age_pedigree
    ]], dtype=float)

    imputed  = IMPUTER.transform(row)
    original = imputed.copy()
    scaled   = SCALER.transform(imputed)

    return scaled, original


CLINICAL_RANGES_CHART = {
    "Glucose":                  {"min": 50,  "max": 200, "normal_lo": 70,  "normal_hi": 99,  "warn_hi": 125, "unit": "mg/dL",    "label": "Blood Sugar (Glucose)"},
    "BMI":                      {"min": 15,  "max": 45,  "normal_lo": 18.5,"normal_hi": 24.9,"warn_hi": 29.9,"unit": "kg/m²",    "label": "Body Mass Index (BMI)"},
    "BloodPressure":            {"min": 40,  "max": 120, "normal_lo": 60,  "normal_hi": 79,  "warn_hi": 89,  "unit": "mmHg",     "label": "Blood Pressure"},
    "Insulin":                  {"min": 0,   "max": 300, "normal_lo": 16,  "normal_hi": 166, "warn_hi": 200, "unit": "mu U/mL",  "label": "Insulin Level"},
    "DiabetesPedigreeFunction": {"min": 0,   "max": 2.5, "normal_lo": 0,   "normal_hi": 0.4, "warn_hi": 0.8, "unit": "score",    "label": "Family History Score"},
    "Age":                      {"min": 18,  "max": 80,  "normal_lo": 18,  "normal_hi": 44,  "warn_hi": 59,  "unit": "years",    "label": "Age"},
    "SkinThickness":            {"min": 0,   "max": 80,  "normal_lo": 10,  "normal_hi": 35,  "warn_hi": 50,  "unit": "mm",       "label": "Skin Thickness"},
    "Pregnancies":              {"min": 0,   "max": 15,  "normal_lo": 0,   "normal_hi": 4,   "warn_hi": 7,   "unit": "count",    "label": "Number of Pregnancies"},
}

def _get_meter_color(value, ref):
    """Return green/amber/red based on where value falls in clinical range."""
    if value <= ref["normal_hi"]:
        return "#16a34a" 
    elif value <= ref["warn_hi"]:
        return "#d97706"   
    else:
        return "#dc2626"   

def _get_meter_status(value, ref):
    if value <= ref["normal_hi"]:   return "Normal"
    elif value <= ref["warn_hi"]:   return "Elevated"
    else:                           return "High"

def generate_health_meter_chart(original: np.ndarray, feature_names: list,
                                shap_vals: np.ndarray, raw_input: dict) -> str:
    """
    Generates a human-friendly 'Health Meter' chart.
    Only shows features where the user actually entered a real value —
    features left as 0 (missing/not entered) are excluded from the chart.

    Thread-safe via PLOT_LOCK. UUID filename per request.
    Returns path relative to /static/ e.g. "shap/meter_abc123.png"
    """
    plot_id   = str(uuid.uuid4())[:10]
    filename  = f"meter_{plot_id}.png"
    save_path = os.path.join(SHAP_PLOT_DIR, filename)

    SKIP_IF_ZERO = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

    rows = []
    for idx in sorted_idx:
        fname    = feature_names[idx]
        raw_val  = raw_input.get(fname, 0)   
        val      = float(original[0][idx])    
        ref      = CLINICAL_RANGES_CHART.get(fname)

        if ref is None:
            continue
        if fname in SKIP_IF_ZERO and raw_val == 0:
            continue

        pct    = min(max((val - ref["min"]) / (ref["max"] - ref["min"]), 0.0), 1.0) * 100
        color  = _get_meter_color(val, ref)
        status = _get_meter_status(val, ref)
        rows.append({
            "label":  ref["label"],
            "value":  val,
            "unit":   ref["unit"],
            "pct":    pct,
            "color":  color,
            "status": status,
            "impact": shap_vals[idx],
        })

    n = len(rows)
    fig_height = max(4.5, n * 0.82 + 1.4)

    with PLOT_LOCK:
        plt.clf()
        fig, ax = plt.subplots(figsize=(9, fig_height))
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, n - 0.5)
        ax.axis("off")
        fig.patch.set_facecolor("#f8fafc")

        BAR_H   = 0.38
        TRACK_H = 0.14
        Y_STEP  = 1.0

        for i, row in enumerate(rows):
            y = (n - 1 - i) * Y_STEP   

            track = plt.Rectangle((0, y - TRACK_H / 2), 100, TRACK_H,
                                   color="#e2e8f0", zorder=1)
            ax.add_patch(track)
   
            ax.axvspan(0,  40,  ymin=(y - TRACK_H/2) / (n*Y_STEP),
                       ymax=(y + TRACK_H/2) / (n*Y_STEP),
                       alpha=0.08, color="#16a34a", zorder=0)
            ax.axvspan(40, 70,  ymin=(y - TRACK_H/2) / (n*Y_STEP),
                       ymax=(y + TRACK_H/2) / (n*Y_STEP),
                       alpha=0.08, color="#d97706", zorder=0)
            ax.axvspan(70, 100, ymin=(y - TRACK_H/2) / (n*Y_STEP),
                       ymax=(y + TRACK_H/2) / (n*Y_STEP),
                       alpha=0.08, color="#dc2626", zorder=0)

            bar_w = max(row["pct"], 1.5)
            bar = plt.Rectangle((0, y - BAR_H / 2), bar_w, BAR_H,
                                  color=row["color"], zorder=2,
                                  linewidth=0, alpha=0.92)
            ax.add_patch(bar)

            circ = plt.Circle((bar_w, y), BAR_H / 2,
                               color=row["color"], zorder=3, alpha=0.92)
            ax.add_patch(circ)
            ax.text(-1.5, y + 0.28, row["label"],
                    ha="right", va="center", fontsize=8.5,
                    color="#334155", fontweight="600",
                    transform=ax.transData, clip_on=False)

            val_x = min(bar_w + 2.5, 100)
            status_color = row["color"]
            ax.text(bar_w + 2.5, y + 0.13,
                    f"{row['value']:.1f} {row['unit']}",
                    ha="left", va="center", fontsize=8.2,
                    color="#1e293b", fontweight="700", zorder=4)
            ax.text(bar_w + 2.5, y - 0.13,
                    row["status"],
                    ha="left", va="center", fontsize=7.2,
                    color=status_color, fontweight="600", zorder=4)

            arrow_color = "#dc2626" if row["impact"] > 0 else "#2563eb"
            arrow_sym   = "▲ Raises risk" if row["impact"] > 0 else "▼ Lowers risk"
            if abs(row["impact"]) > 0.01:
                ax.text(96, y, arrow_sym,
                        ha="right", va="center", fontsize=6.8,
                        color=arrow_color, fontweight="700", zorder=4)

   
        legend_y = -0.55
        for lx, lc, lt in [(2, "#16a34a", "● Normal"),
                            (18, "#d97706", "● Elevated"),
                            (34, "#dc2626", "● High / Concerning")]:
            ax.text(lx, legend_y, lt, ha="left", va="center",
                    fontsize=7.5, color=lc, fontweight="600")

        ax.text(98, legend_y, "▲▼ = Effect on your diabetes risk",
                ha="right", va="center", fontsize=7, color="#64748b")

  
        fig.suptitle("Your Health Indicators at a Glance",
                     fontsize=11, fontweight="bold", color="#1e293b",
                     x=0.55, y=0.98)

        plt.tight_layout(rect=[0.18, 0.04, 1.0, 0.96])
        plt.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="#f8fafc")
        plt.close(fig)

    return f"shap/{filename}"


def build_gemini_prompt(prob: float, tier_label: str, tier_key: str, top_features: list) -> str:
    """
    Builds a risk-tier-specific prompt so Gemini gives different
    recommendations for LOW / MODERATE / HIGH risk patients.
    """
    top3 = top_features[:3]
    driver_lines = "\n".join([
        f"  {i+1}. {f['feature']} = {f['value_display']} "
        f"[Status: {f['status']}] "
        f"| {'INCREASES' if f['shap'] > 0 else 'REDUCES'} risk "
        f"(SHAP contribution: {abs(f['shap']):.4f})"
        for i, f in enumerate(top3)
    ])

    if tier_key == "high":
        tier_guidance = """
RISK TIER: HIGH (≥70% probability)
Tone: Urgent but calm. This patient needs prompt medical attention.
- risk_summary: Explain clearly that their values are in a concerning range and they should see a doctor soon. Do NOT cause panic. Reference their specific glucose/BMI values.
- lifestyle_advice: Give urgent, specific changes they must start TODAY. Be direct.
- follow_up_tests: Include at least one test marked "Immediate (within 48hrs)". Recommend HbA1c and fasting glucose as priority.
- urgency options available: "Immediate (within 48hrs)", "Within 2 weeks", "Within 1 month", "Routine annual"
"""
    elif tier_key == "moderate":
        tier_guidance = """
RISK TIER: MODERATE (40–69% probability)
Tone: Cautiously encouraging. This patient has warning signs but has time to act.
- risk_summary: Explain they are at an elevated but not critical risk. Emphasise that lifestyle changes NOW can meaningfully reduce their risk. Reference specific values.
- lifestyle_advice: Give practical, achievable changes. Mention small wins and gradual improvement.
- follow_up_tests: Include tests marked "Within 2 weeks" or "Within 1 month". No "Immediate" tests needed unless a specific value warrants it.
- urgency options available: "Immediate (within 48hrs)", "Within 2 weeks", "Within 1 month", "Routine annual"
"""
    else:  
        tier_guidance = """
RISK TIER: LOW (<40% probability)
Tone: Positive and motivating. This patient is doing well — reinforce and maintain.
- risk_summary: Reassure them their current indicators look good. Encourage continued healthy habits. Mention what specific values are working in their favour.
- lifestyle_advice: Give maintenance-focused advice. Emphasise prevention and sustainability. Avoid alarmist language.
- follow_up_tests: Mostly "Routine annual" tests. Only include 2-3 tests. No urgent tests.
- urgency options available: "Immediate (within 48hrs)", "Within 2 weeks", "Within 1 month", "Routine annual"
"""

    return f"""You are a clinical decision support assistant for early diabetes detection.

=== PATIENT RISK ASSESSMENT ===
Risk Level        : {tier_label}
Risk Probability  : {prob:.1%}

=== TOP 3 FACTORS THAT INFLUENCED THIS RESULT ===
{driver_lines}

=== TIER-SPECIFIC GUIDANCE ===
{tier_guidance}

=== OUTPUT INSTRUCTIONS ===
Return ONLY a valid JSON object. No markdown. No code fences. No text outside the JSON.

Exact structure:
{{
  "risk_summary": "3-4 sentences written directly to the patient. Reference their actual values. Plain, non-medical language appropriate to their risk tier.",

  "lifestyle_advice": [
    {{
      "category": "Diet",
      "icon": "restaurant",
      "advice": "Tier-appropriate dietary advice that references this patient's actual glucose or BMI value."
    }},
    {{
      "category": "Physical Activity",
      "icon": "directions_run",
      "advice": "Tier-appropriate exercise recommendation. HIGH = specific urgent target. LOW = maintenance."
    }},
    {{
      "category": "Weight Management",
      "icon": "monitor_weight",
      "advice": "Based on this patient's actual BMI value. HIGH/MODERATE = reduction goal. LOW = maintain."
    }},
    {{
      "category": "Blood Sugar Monitoring",
      "icon": "water_drop",
      "advice": "HIGH = daily monitoring. MODERATE = weekly. LOW = monthly or at routine check-up."
    }}
  ],

  "follow_up_tests": [
    {{
      "test": "Test name",
      "urgency": "One of: Immediate (within 48hrs) / Within 2 weeks / Within 1 month / Routine annual",
      "reason": "One sentence explaining why for this specific patient."
    }}
  ]
}}

Hard rules:
- lifestyle_advice: exactly 4 items in the order shown above
- follow_up_tests: HIGH tier = 3-4 tests, MODERATE = 2-3 tests, LOW = 2 tests
- Advice must reference the patient's actual values — not generic text
- Urgency must exactly match one of the 4 specified options"""


def call_gemini(prompt: str) -> dict:
    """
    Calls Gemini with 3 retries. Always returns a valid dict.
    If Gemini fails entirely, returns safe fallback recommendations
    so the DSS shows ML prediction + SHAP even without LLM output.
    """
    FALLBACK = {
        "risk_summary": (
            "AI summary is temporarily unavailable. "
            "Your diabetes risk score and SHAP feature analysis above are still accurate. "
            "Please consult a qualified healthcare professional to review these results."
        ),
        "lifestyle_advice": [
            {"category": "Diet",                   "icon": "restaurant",     "advice": "Reduce refined sugars and processed carbohydrates. Increase vegetables, legumes, and whole grains."},
            {"category": "Physical Activity",      "icon": "directions_run", "advice": "Aim for 150 minutes of moderate exercise per week such as brisk walking or swimming."},
            {"category": "Weight Management",      "icon": "monitor_weight", "advice": "If your BMI is above 25, a 5-10% weight reduction significantly lowers diabetes risk."},
            {"category": "Blood Sugar Monitoring", "icon": "water_drop",     "advice": "Monitor fasting glucose regularly. Keep a log to share with your doctor at your next visit."},
        ],
        "follow_up_tests": [
            {"test": "Fasting Blood Glucose",       "urgency": "Within 2 weeks",  "reason": "Confirms glucose status with a controlled fasting measurement."},
            {"test": "HbA1c",                       "urgency": "Within 1 month",  "reason": "Reflects average blood sugar levels over the past 3 months."},
            {"test": "Lipid Profile",               "urgency": "Within 1 month",  "reason": "Diabetes and cardiovascular disease risk are strongly correlated."},
        ],
        "_gemini_failed": True
    }

    for attempt in range(1, 4):
        try:
            response = genai.GenerativeModel(GEMINI_MODEL).generate_content(prompt)
            raw      = response.text.strip()

            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE)
            raw = raw.strip()

            result = json.loads(raw)
            print(f"[INFO] Gemini OK (attempt {attempt})")
            return result

        except json.JSONDecodeError as e:
            print(f"[WARN] Attempt {attempt} JSON error: {e}")
            time.sleep(1.5)
        except Exception as e:
            print(f"[WARN] Attempt {attempt} Gemini error: {e}")
            time.sleep(2)

    print("[WARN] Gemini failed all attempts — using fallback.")
    return FALLBACK


@app.route("/")
def index():
    return render_template("ui.html")


@app.route("/predict", methods=["POST"])
def predict():
    def get_float(name, default=0.0):
        val = request.form.get(name, "").strip()
        try:    return float(val) if val else default
        except: return default

    raw_input = {
        "Pregnancies":              get_float("pregnancies",   0.0),
        "Glucose":                  get_float("glucose",       0.0),
        "BloodPressure":            get_float("bloodpressure", 0.0),
        "SkinThickness":            get_float("skinthickness", 0.0),
        "Insulin":                  get_float("insulin",       0.0),
        "BMI":                      get_float("bmi",           0.0),
        "DiabetesPedigreeFunction": get_float("pedigree",      0.471),
        "Age":                      get_float("age",           0.0),
    }

    scaled, original = preprocess_input(raw_input)

    prob       = float(MODEL.predict_proba(scaled)[0][1])
    predicted  = int(prob >= THRESHOLD)
    tier_label, tier_key = get_risk_tier(prob)
    confidence = get_confidence_label(prob)

    shap_vals  = EXPLAINER.shap_values(scaled)[0]
    total_abs  = float(np.sum(np.abs(shap_vals))) + 1e-9

    top_idx      = np.argsort(np.abs(shap_vals))[::-1][:5]
    top_features = []
    for idx in top_idx:
        fname    = FEATURE_NAMES[idx]
        orig_val = float(original[0][idx])
        unit     = CLINICAL_UNITS.get(fname, "")
        top_features.append({
            "feature":       fname,
            "value":         round(orig_val, 1),
            "value_display": f"{orig_val:.1f} {unit}".strip(),
            "unit":          unit,
            "status":        get_clinical_status(fname, orig_val),
            "shap":          round(float(shap_vals[idx]), 4),
            "shap_pct":      round(abs(float(shap_vals[idx])) / total_abs * 100, 1),
        })

    all_features = []
    for i, feat in enumerate(FEATURE_NAMES):
        orig_val = float(original[0][i])
        all_features.append({
            "feature": feat,
            "value":   round(orig_val, 1),
            "unit":    CLINICAL_UNITS.get(feat, ""),
            "status":  get_clinical_status(feat, orig_val),
            "shap":    round(float(shap_vals[i]), 4),
        })

    shap_plot_url = generate_health_meter_chart(original, FEATURE_NAMES, shap_vals, raw_input)

    try:
        prompt       = build_gemini_prompt(prob, tier_label, tier_key, top_features)
        llm_response = call_gemini(prompt)
    except Exception as e:
        print(f"[ERROR] Gemini outer error: {e}")
        llm_response = {
            "risk_summary":    "AI summary unavailable. ML prediction and SHAP results are shown above.",
            "lifestyle_advice": [],
            "follow_up_tests":  [],
            "_gemini_failed":   True
        }

    gemini_failed = llm_response.pop("_gemini_failed", False)

    session["result"] = {
        "probability":   round(prob * 100, 1),
        "prob_raw":      round(prob, 4),
        "predicted":     predicted,
        "tier_label":    tier_label,
        "tier_key":      tier_key,
        "confidence":    confidence,
        "top_features":  top_features,
        "all_features":  all_features,
        "shap_plot_url": shap_plot_url,
        "risk_summary":  llm_response.get("risk_summary", ""),
        "lifestyle":     llm_response.get("lifestyle_advice", []),
        "tests":         llm_response.get("follow_up_tests", []),
        "gemini_failed": gemini_failed,
        "input":         {k: round(float(v), 1) for k, v in raw_input.items()},
    }

    return redirect(url_for("results"))


@app.route("/results")
def results():
    data = session.get("result")
    if not data:
        return redirect(url_for("index"))
    return render_template("risk.html", data=data)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Gemini-powered context-aware chatbot endpoint.
    Receives: { message: str, context: { prob, tier, tierLabel, glucose, bmi, bp, age, insulin, pedigree } }
    Returns:  { reply: str }
    """
    try:
        api_key = GEMINI_API_KEY
        genai.configure(api_key=api_key)

        payload = request.get_json(force=True)
        user_msg = payload.get("message", "").strip()
        ctx      = payload.get("context", {})

        if not user_msg:
            return jsonify({"reply": "Please type a question."})

        tier = str(ctx.get("tier", "low") or "low")
        prob = float(ctx.get("prob", 0) or 0)

        if tier == "high":
            system_persona = (
                "You are a compassionate but clinically urgent AI health advisor. "
                "This patient has a HIGH diabetes risk. Be clear, direct, and empathetic. "
                "Emphasise the importance of prompt medical attention without causing panic."
            )
        elif tier == "moderate":
            system_persona = (
                "You are an encouraging AI health advisor. "
                "This patient is at MODERATE diabetes risk — they have time to act. "
                "Be motivational and practical. Emphasise that lifestyle changes now can "
                "meaningfully reduce their risk."
            )
        else:
            system_persona = (
                "You are a positive and supportive AI health advisor. "
                "This patient is at LOW diabetes risk — their indicators are good. "
                "Reinforce healthy habits and provide preventive guidance."
            )

        glucose    = float(ctx.get("glucose",  0) or 0)
        bmi        = float(ctx.get("bmi",       0) or 0)
        bp         = float(ctx.get("bp",        0) or 0)
        age        = float(ctx.get("age",       0) or 0)
        insulin    = float(ctx.get("insulin",   0) or 0)
        pedigree   = float(ctx.get("pedigree",  0) or 0)
        top_feat   = ctx.get("topFeature", "Glucose")
        top_status = ctx.get("topStatus",  "Unknown")

        patient_block = f"""
=== THIS PATIENT'S SPECIFIC HEALTH DATA ===
Diabetes Risk Probability : {prob}% ({tier.upper()} RISK)
Blood Glucose             : {glucose} mg/dL  {'[Pre-diabetic]' if glucose >= 100 else '[Normal]' if glucose >= 70 else ''}
BMI                       : {bmi}  {'[Obese]' if bmi >= 30 else '[Overweight]' if bmi >= 25 else '[Healthy]'}
Blood Pressure (diastolic): {bp} mmHg  {'[Hypertensive]' if bp >= 90 else '[Normal]'}
Age                       : {age} years
Insulin                   : {insulin} mu U/mL
Family History Score      : {pedigree}
Strongest Risk Factor     : {top_feat} ({top_status})
"""

        chat_prompt = (
            f"{system_persona}\n\n"
            f"{patient_block}\n"
            "=== INSTRUCTIONS ===\n"
            "The patient is asking a question about their health results. "
            "Reference their SPECIFIC values above (e.g., their exact glucose of "
            f"{glucose} mg/dL, BMI of {bmi}) in your answer — never give generic advice. "
            "Keep the response under 5 sentences, warm, and jargon-free. "
            "Do NOT recommend specific medications. "
            "Always suggest they consult a qualified healthcare professional for diagnosis.\n\n"
            f"=== PATIENT QUESTION ===\n{user_msg}"
        )

        for attempt in range(1, 4):
            try:
                response = genai.GenerativeModel(GEMINI_MODEL).generate_content(chat_prompt)
                reply    = response.text.strip()
                if reply:
                    return jsonify({"reply": reply})
            except Exception as e:
                print(f"[WARN] Chat Gemini attempt {attempt} error: {e}")
                time.sleep(1.5)

        fallback = (
            f"Based on your {prob}% risk ({tier.upper()} tier), "
            f"your {top_feat} ({top_status}) is your most influential factor. "
            "Please consult a qualified healthcare professional for personalised advice."
        )
        return jsonify({"reply": fallback})

    except Exception as e:
        print(f"[ERROR] /chat route error: {e}")
        return jsonify({"reply": "Sorry, I encountered an error. Please try again."}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload   = request.get_json(force=True)
        raw_input = {
            "Pregnancies":              float(payload.get("pregnancies",   0)),
            "Glucose":                  float(payload.get("glucose",       0)),
            "BloodPressure":            float(payload.get("bloodpressure", 0)),
            "SkinThickness":            float(payload.get("skinthickness", 0)),
            "Insulin":                  float(payload.get("insulin",       0)),
            "BMI":                      float(payload.get("bmi",           0)),
            "DiabetesPedigreeFunction": float(payload.get("pedigree",      0.471)),
            "Age":                      float(payload.get("age",           0)),
        }
        scaled, original = preprocess_input(raw_input)
        prob      = float(MODEL.predict_proba(scaled)[0][1])
        predicted = int(prob >= THRESHOLD)
        tier_label, tier_key = get_risk_tier(prob)
        shap_vals = EXPLAINER.shap_values(scaled)[0]
        top_idx   = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_features = [
            {
                "feature": FEATURE_NAMES[i],
                "value":   round(float(original[0][i]), 1),
                "unit":    CLINICAL_UNITS.get(FEATURE_NAMES[i], ""),
                "shap":    round(float(shap_vals[i]), 4),
                "status":  get_clinical_status(FEATURE_NAMES[i], float(original[0][i]))
            }
            for i in top_idx
        ]
        return jsonify({
            "status":       "success",
            "probability":  round(prob, 4),
            "predicted":    predicted,
            "tier":         tier_label,
            "tier_key":     tier_key,
            "confidence":   get_confidence_label(prob),
            "top_features": top_features
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  HealthAI DSS — Flask (Refined v3)")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 55 + "\n")
    app.run(debug=True, port=5000, threaded=True)