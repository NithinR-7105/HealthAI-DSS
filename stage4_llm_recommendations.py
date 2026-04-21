import os
import re
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

import google.generativeai as genai

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "\n[ERROR] GEMINI_API_KEY environment variable is not set.\n"
        "Set it before running:\n"
        "  Windows : set GEMINI_API_KEY=your_key_here\n"
        "  Mac/Linux: export GEMINI_API_KEY=your_key_here\n"
    )

DATA_PATH    = "C:/Users/nithi/Desktop/diabetes_env/diabetes.csv"
OUTPUT_DIR   = "results"
MODEL_DIR    = "model"
NUM_PATIENTS = 5 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

CLINICAL_RANGES = {
    "Glucose": {
        "unit": "mg/dL",
        "thresholds": {"normal": (70, 99), "prediabetes": (100, 125), "diabetic": 126}
    },
    "BloodPressure": {
        "unit": "mmHg",
        "thresholds": {"normal": (60, 80), "high": 90}
    },
    "SkinThickness": {"unit": "mm", "thresholds": {"normal": (10, 25)}},
    "Insulin":       {"unit": "mu U/mL", "thresholds": {"normal": (16, 166)}},
    "BMI": {
        "unit": "kg/m2",
        "thresholds": {"normal": (18.5, 24.9), "overweight": (25, 29.9), "obese": 30}
    },
    "DiabetesPedigreeFunction": {"unit": "score", "thresholds": {}},
    "Age":         {"unit": "years", "thresholds": {}},
    "Pregnancies": {"unit": "count", "thresholds": {}},
    "Glucose_BMI": {"unit": "composite", "thresholds": {},
                    "note": "Interaction: high glucose × high BMI compounds risk"},
    "Age_Pedigree":{"unit": "composite", "thresholds": {},
                    "note": "Interaction: older age × strong family history"}
}


def get_clinical_status(feature, value):
    """Return a plain-English clinical status for a feature value."""
    if feature == "Glucose":
        if value >= 126:   return "Diabetic range"
        elif value >= 100: return "Pre-diabetic (elevated)"
        elif value >= 70:  return "Normal"
        else:              return "Low (hypoglycaemia risk)"
    elif feature == "BMI":
        if value >= 30:    return "Obese"
        elif value >= 25:  return "Overweight"
        elif value >= 18.5:return "Normal weight"
        else:              return "Underweight"
    elif feature == "BloodPressure":
        if value >= 90:    return "Hypertensive"
        elif value >= 60:  return "Normal"
        else:              return "Low"
    elif feature == "DiabetesPedigreeFunction":
        if value > 0.8:    return "Strong family history"
        elif value > 0.4:  return "Moderate family history"
        else:              return "Low family history"
    elif feature == "Age":
        if value >= 60:    return "Senior"
        elif value >= 45:  return "Middle-aged"
        else:              return "Younger adult"
    elif feature == "Glucose_BMI":
        if value > 5000:   return "HIGH combined risk (elevated glucose + high BMI)"
        elif value > 3000: return "MODERATE combined risk"
        else:              return "Lower combined risk"
    elif feature == "Age_Pedigree":
        if value > 30:     return "HIGH compounded risk (older age + strong family history)"
        elif value > 15:   return "MODERATE compounded risk"
        else:              return "Lower compounded risk"
    return "—"


def get_risk_tier(prob):
    """Convert XGBoost probability to a named risk tier."""
    if prob >= 0.70:
        return "HIGH RISK", "#C0392B"
    elif prob >= 0.40:
        return "MODERATE RISK", "#E67E22"
    else:
        return "LOW RISK", "#27AE60"

def load_or_train_model(data_path):
    model_path   = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    imputer_path = os.path.join(MODEL_DIR, "imputer.pkl")

    if all(os.path.exists(p) for p in [model_path, scaler_path, imputer_path]):
        print("[INFO] Loading saved model from Stage 3 ...")
        model   = joblib.load(model_path)
        scaler  = joblib.load(scaler_path)
        imputer = joblib.load(imputer_path)
    else:
        print("[INFO] Saved model not found. Retraining with stage2 configuration ...")
        df = pd.read_csv(data_path)
        invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]
        X = X.copy()
        X["Glucose_BMI"]  = X["Glucose"].fillna(X["Glucose"].median()) * \
                            X["BMI"].fillna(X["BMI"].median())
        X["Age_Pedigree"] = X["Age"] * X["DiabetesPedigreeFunction"]

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        imputer = SimpleImputer(strategy="median")
        scaler  = StandardScaler()
        X_tr_imp = imputer.fit_transform(X_train)
        X_tr_sc  = scaler.fit_transform(X_tr_imp)
        smote_tomek = SMOTETomek(random_state=42)
        X_res, y_res = smote_tomek.fit_resample(X_tr_sc, y_train)

        from sklearn.model_selection import train_test_split as tts_inner
        X_tr2, X_val2, y_tr2, y_val2 = tts_inner(
            X_res, y_res, test_size=0.1, random_state=42
        )
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            min_child_weight=2, gamma=0.0,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", early_stopping_rounds=30,
            random_state=42, verbosity=0
        )
        model.fit(
            X_tr2, y_tr2,
            eval_set=[(X_tr2, y_tr2), (X_val2, y_val2)],
            verbose=False
        )

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(imputer, imputer_path)
        print("[INFO] Model retrained and saved.")

    return model, scaler, imputer

def prepare_test_data(data_path, imputer, scaler):
    df = pd.read_csv(data_path)
    invalid_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[invalid_cols] = df[invalid_cols].replace(0, np.nan)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X = X.copy()
    X["Glucose_BMI"]  = X["Glucose"].fillna(X["Glucose"].median()) * \
                        X["BMI"].fillna(X["BMI"].median())
    X["Age_Pedigree"] = X["Age"] * X["DiabetesPedigreeFunction"]

    feature_names = list(X.columns)  

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_test = y_test.reset_index(drop=True)

    X_test_imp      = imputer.transform(X_test)
    X_test_original = X_test_imp.copy()          
    X_test_scaled   = scaler.transform(X_test_imp)

    return X_test_scaled, X_test_original, y_test, feature_names

def build_gemini_prompt(patient_data: dict) -> str:
    """
    Converts structured patient SHAP data into a precise clinical prompt
    that instructs Gemini to return a JSON response with three sections.
    """
    feature_lines = "\n".join([
        f"  - {f['feature']}: {f['value']:.1f} {f['unit']} "
        f"[{f['status']}] | SHAP impact: {'↑ INCREASES risk' if f['shap'] > 0 else '↓ REDUCES risk'} "
        f"(SHAP={f['shap']:+.4f})"
        for f in patient_data["top_features"]
    ])

    prompt = f"""
You are a clinical decision support assistant specialising in early diabetes detection.
A machine learning model (XGBoost with SHAP explainability) has assessed a patient's
diabetes risk. Your task is to synthesise this into a clear, actionable clinical report.

--- PATIENT DATA ---
Risk Probability  : {patient_data['probability']:.1%}
Risk Tier         : {patient_data['risk_tier']}
Prediction        : {"DIABETIC" if patient_data['predicted'] == 1 else "NON-DIABETIC"}

Top Clinical Risk Drivers (SHAP Analysis):
{feature_lines}

--- YOUR TASK ---
Generate a structured clinical recommendation report in VALID JSON format only.
Do NOT include markdown, code blocks, or any text outside the JSON.

Return exactly this JSON structure:
{{
  "risk_summary": "A 3-4 sentence natural language paragraph summarising this patient's
                   overall diabetes risk, referencing their specific clinical values and
                   explaining which factors are most concerning and why.",

  "lifestyle_advice": [
    {{
      "category": "Diet",
      "advice": "Specific, actionable dietary recommendation based on this patient's data."
    }},
    {{
      "category": "Physical Activity",
      "advice": "Specific exercise recommendation tailored to this patient's risk profile."
    }},
    {{
      "category": "Weight Management",
      "advice": "Specific recommendation if BMI is a risk factor, else general guidance."
    }},
    {{
      "category": "Blood Sugar Monitoring",
      "advice": "Home monitoring advice appropriate to this patient's risk tier."
    }}
  ],

  "follow_up_tests": [
    {{
      "test": "Test name",
      "urgency": "Immediate / Within 2 weeks / Within 1 month / Routine annual",
      "reason": "Why this specific test is recommended for this patient."
    }}
  ]
}}

Be medically precise. Tailor every recommendation to this specific patient's values.
Do not use generic advice — reference actual feature values from the data above.
"""
    return prompt.strip()

def call_gemini(prompt: str, retries: int = 3) -> dict:
    """Send prompt to Gemini and parse JSON response."""
    for attempt in range(1, retries + 1):
        try:
            response = gemini_model.generate_content(prompt)
            raw_text = response.text.strip()
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)

            return json.loads(raw_text)

        except json.JSONDecodeError as e:
            print(f"  [WARN] Attempt {attempt}: JSON parse error — {e}. Retrying ...")
            time.sleep(2)
        except Exception as e:
            print(f"  [ERROR] Gemini API error on attempt {attempt}: {e}")
            time.sleep(3)

    return {
        "risk_summary": "Unable to generate LLM summary. Please check your API key and retry.",
        "lifestyle_advice": [],
        "follow_up_tests": []
    }

def print_patient_report(patient_idx, patient_data, llm_response):
    tier, _ = get_risk_tier(patient_data["probability"])
    print(f"\n{'='*65}")
    print(f"  PATIENT #{patient_idx + 1} — CLINICAL DECISION SUPPORT REPORT")
    print(f"{'='*65}")
    print(f"  Risk Tier         : {tier}")
    print(f"  Risk Probability  : {patient_data['probability']:.2%}")
    print(f"  Prediction        : {'DIABETIC' if patient_data['predicted'] == 1 else 'NON-DIABETIC'}")
    print(f"  True Outcome      : {'DIABETIC' if patient_data['true_label'] == 1 else 'NON-DIABETIC'}")

    print(f"\n  --- TOP SHAP RISK DRIVERS ---")
    for f in patient_data["top_features"]:
        direction = "↑" if f["shap"] > 0 else "↓"
        print(f"    {direction} {f['feature']:<32} {f['value']:.1f} {f['unit']:<12} [{f['status']}]")

    print(f"\n  --- GEMINI: RISK SUMMARY ---")
    summary = llm_response.get("risk_summary", "N/A")
    # Word-wrap to 60 chars
    words = summary.split()
    line, out = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 60:
            out.append("  " + " ".join(line[:-1]))
            line = [w]
    if line:
        out.append("  " + " ".join(line))
    print("\n".join(out))

    print(f"\n  --- GEMINI: LIFESTYLE ADVICE ---")
    for item in llm_response.get("lifestyle_advice", []):
        print(f"  [{item.get('category','—')}]")
        print(f"    {item.get('advice','—')}")

    print(f"\n  --- GEMINI: RECOMMENDED FOLLOW-UP TESTS ---")
    for test in llm_response.get("follow_up_tests", []):
        print(f"  • {test.get('test','—')} [{test.get('urgency','—')}]")
        print(f"    Reason: {test.get('reason','—')}")

    print(f"{'='*65}\n")


def save_shap_waterfall(explainer, shap_vals, patient_idx, feature_names, save_path):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals   = shap_vals[patient_idx]
        sorted_idx = np.argsort(np.abs(vals))[::-1][:8]
        top_features = [feature_names[i] for i in sorted_idx]
        top_vals     = [vals[i] for i in sorted_idx]

        bar_colors = ["#C0392B" if v > 0 else "#2980B9" for v in top_vals]
        y_pos = range(len(top_features))
        ax.barh(list(y_pos), top_vals[::-1], color=bar_colors[::-1], edgecolor="white")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(top_features[::-1], fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Patient #{patient_idx + 1} — SHAP Feature Impact", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        print(f"  [WARN] Could not save SHAP plot: {e}")
        return False


def generate_pdf_report(all_patient_reports: list, output_path: str):
    """Generate a multi-patient clinical PDF report using ReportLab."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()


    title_style = ParagraphStyle(
        "ClinicalTitle",
        parent=styles["Title"],
        fontSize=18,
        textColor=colors.HexColor("#1A2E4A"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#555555"),
        spaceAfter=16,
        alignment=TA_CENTER
    )
    section_style = ParagraphStyle(
        "SectionHead",
        parent=styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#1A2E4A"),
        spaceBefore=12,
        spaceAfter=4,
        borderPad=4
    )
    body_style = ParagraphStyle(
        "ClinicalBody",
        parent=styles["Normal"],
        fontSize=9,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.HexColor("#222222")
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#444444"),
        leading=12
    )

    RISK_COLORS = {
        "HIGH RISK":     colors.HexColor("#C0392B"),
        "MODERATE RISK": colors.HexColor("#E67E22"),
        "LOW RISK":      colors.HexColor("#27AE60")
    }

    story = []

    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("Diabetic Early Stage Detection", title_style))
    story.append(Paragraph("Clinical Decision Support Report — Stage 4", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5,
                             color=colors.HexColor("#1A2E4A"), spaceAfter=10))
    story.append(Paragraph(
        f"Generated by XGBoost + SHAP + Google Gemini Pipeline &nbsp;|&nbsp; "
        f"Patients Analysed: {len(all_patient_reports)}",
        subtitle_style
    ))
    story.append(Spacer(1, 0.5*cm))

    # Summary table
    summary_data = [["#", "Risk Tier", "Probability", "Prediction", "True Outcome"]]
    for rep in all_patient_reports:
        tier = rep["patient_data"]["risk_tier"]
        summary_data.append([
            str(rep["patient_idx"] + 1),
            tier,
            f"{rep['patient_data']['probability']:.1%}",
            "Diabetic" if rep["patient_data"]["predicted"] == 1 else "Non-Diabetic",
            "Diabetic" if rep["patient_data"]["true_label"] == 1 else "Non-Diabetic"
        ])

    tbl = Table(summary_data, colWidths=[1.2*cm, 4*cm, 3*cm, 3.5*cm, 3.5*cm])
    tbl_style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#1A2E4A")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#F2F2F2"), colors.white]),
        ("GRID",        (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ])
 
    for i, rep in enumerate(all_patient_reports, start=1):
        tier = rep["patient_data"]["risk_tier"]
        tbl_style.add("TEXTCOLOR", (1, i), (1, i), RISK_COLORS.get(tier, colors.black))
        tbl_style.add("FONTNAME",  (1, i), (1, i), "Helvetica-Bold")

    tbl.setStyle(tbl_style)
    story.append(tbl)
    story.append(PageBreak())

    for rep in all_patient_reports:
        pid       = rep["patient_idx"] + 1
        pd_data   = rep["patient_data"]
        llm       = rep["llm_response"]
        tier      = pd_data["risk_tier"]
        tier_color = RISK_COLORS.get(tier, colors.black)
        shap_img  = rep.get("shap_img_path")


        header_style = ParagraphStyle(
            "PatHeader", parent=styles["Normal"],
            fontSize=13, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1A2E4A"), spaceAfter=2
        )
        story.append(Paragraph(f"Patient #{pid} — Clinical Risk Report", header_style))


        badge_style = ParagraphStyle(
            "Badge", parent=styles["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=tier_color, spaceAfter=8
        )
        story.append(Paragraph(
            f"{tier}   |   Probability: {pd_data['probability']:.1%}   |   "
            f"Predicted: {'Diabetic' if pd_data['predicted'] == 1 else 'Non-Diabetic'}   |   "
            f"Actual: {'Diabetic' if pd_data['true_label'] == 1 else 'Non-Diabetic'}",
            badge_style
        ))
        story.append(HRFlowable(width="100%", thickness=0.8,
                                 color=colors.HexColor("#CCCCCC"), spaceAfter=8))

        story.append(Paragraph("Key Risk Drivers (SHAP Analysis)", section_style))
        feat_data = [["Feature", "Value", "Status", "Direction", "SHAP"]]
        for f in pd_data["top_features"]:
            direction = "↑ Increases risk" if f["shap"] > 0 else "↓ Reduces risk"
            feat_data.append([
                f["feature"],
                f"{f['value']:.1f} {f['unit']}",
                f["status"],
                direction,
                f"{f['shap']:+.4f}"
            ])

        feat_tbl = Table(feat_data, colWidths=[4*cm, 3*cm, 3.5*cm, 3.8*cm, 2.2*cm])
        feat_style = TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#FAFAFA"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
            ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ])
        for i, f in enumerate(pd_data["top_features"], start=1):
            c = colors.HexColor("#C0392B") if f["shap"] > 0 else colors.HexColor("#2980B9")
            feat_style.add("TEXTCOLOR", (3, i), (3, i), c)
            feat_style.add("FONTNAME",  (3, i), (3, i), "Helvetica-Bold")
        feat_tbl.setStyle(feat_style)
        story.append(feat_tbl)
        story.append(Spacer(1, 0.3*cm))


        if shap_img and os.path.exists(shap_img):
            story.append(RLImage(shap_img, width=14*cm, height=6*cm))
            story.append(Spacer(1, 0.3*cm))

        story.append(Paragraph("AI-Generated Risk Summary (Google Gemini)", section_style))
        summary_text = llm.get("risk_summary", "Not available.")
        story.append(Paragraph(summary_text, body_style))
        story.append(Spacer(1, 0.3*cm))
        lifestyle = llm.get("lifestyle_advice", [])
        if lifestyle:
            story.append(Paragraph("Lifestyle Recommendations", section_style))
            for item in lifestyle:
                cat    = item.get("category", "—")
                advice = item.get("advice", "—")
                story.append(Paragraph(
                    f"<b>{cat}:</b> {advice}", small_style
                ))
                story.append(Spacer(1, 0.15*cm))

        tests = llm.get("follow_up_tests", [])
        if tests:
            story.append(Paragraph("Recommended Follow-Up Tests", section_style))
            test_data = [["Test", "Urgency", "Reason"]]
            for t in tests:
                test_data.append([
                    t.get("test", "—"),
                    t.get("urgency", "—"),
                    t.get("reason", "—")
                ])
            test_tbl = Table(test_data, colWidths=[4*cm, 3.5*cm, 9*cm])
            test_tbl.setStyle(TableStyle([
                ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
                ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#FAFAFA"), colors.white]),
                ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#DDDDDD")),
                ("VALIGN",       (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING",   (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
                ("LEFTPADDING",  (0, 0), (-1, -1), 5),
            ]))
            story.append(test_tbl)

        # Disclaimer
        story.append(Spacer(1, 0.4*cm))
        disclaimer_style = ParagraphStyle(
            "Disclaimer", parent=styles["Normal"],
            fontSize=7, textColor=colors.HexColor("#888888"),
            borderColor=colors.HexColor("#DDDDDD"),
            borderWidth=0.5, borderPad=4, leading=10
        )
        story.append(Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI-based decision support system "
            "and is intended to assist, not replace, clinical judgment. All recommendations must "
            "be reviewed and validated by a qualified healthcare professional before action.",
            disclaimer_style
        ))
        story.append(PageBreak())

    doc.build(story)
    print(f"[INFO] PDF report saved: {output_path}")


def generate_streamlit_app():
    app_code = '''# ============================================================
# Stage 4 — Streamlit DSS Interface
# Run with: streamlit run streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import re
import os
import shap
import matplotlib.pyplot as plt
import google.generativeai as genai
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic DSS",
    page_icon="🩺",
    layout="wide"
)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
DATA_PATH      = "diabetes.csv"
MODEL_DIR      = "model"

RISK_COLORS = {
    "HIGH RISK":     "#C0392B",
    "MODERATE RISK": "#E67E22",
    "LOW RISK":      "#27AE60"
}

@st.cache_resource
def load_artifacts():
    model   = joblib.load(f"{MODEL_DIR}/model.pkl")
    scaler  = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    imputer = joblib.load(f"{MODEL_DIR}/imputer.pkl")
    return model, scaler, imputer

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    invalid = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[invalid] = df[invalid].replace(0, np.nan)
    return df

def get_risk_tier(prob):
    if prob >= 0.70: return "HIGH RISK"
    elif prob >= 0.40: return "MODERATE RISK"
    return "LOW RISK"

def call_gemini(prompt_text, api_key):
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel("gemini-2.5-flash")
    response = m.generate_content(prompt_text)
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\\s*", "", raw)
    raw = re.sub(r"\\s*```$", "", raw)
    return json.loads(raw)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/stethoscope.png", width=60)
    st.title("🩺 Diabetic DSS")
    st.caption("Early Stage Detection & Clinical Decision Support")
    st.divider()
    api_key = st.text_input("Google Gemini API Key", value=GEMINI_API_KEY, type="password")
    st.divider()
    st.markdown("**Manual Patient Input**")
    use_manual = st.checkbox("Enter patient data manually")

# ── Main ────────────────────────────────────────────────────
st.title("Diabetic Early Stage Detection")
st.subheader("Clinical Decision Support System — Stage 4")
st.divider()

model, scaler, imputer = load_artifacts()
df = load_data()

feature_names = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age",
                  "Glucose_BMI","Age_Pedigree"]

if use_manual:
    st.markdown("### Enter Patient Clinical Data")
    cols = st.columns(4)
    inputs = {}
    base_feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                  "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    defaults = {"Pregnancies":2,"Glucose":120,"BloodPressure":70,
                "SkinThickness":20,"Insulin":80,"BMI":27.0,
                "DiabetesPedigreeFunction":0.5,"Age":35}
    for i, feat in enumerate(base_feats):
        with cols[i % 4]:
            inputs[feat] = st.number_input(feat, value=float(defaults[feat]), step=0.1)

    if st.button("🔍 Analyse Patient", type="primary"):
        base_feats = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                      "Insulin","BMI","DiabetesPedigreeFunction","Age"]
        base_vals  = np.array([[inputs[f] for f in base_feats]])
        glucose_bmi   = inputs["Glucose"] * inputs["BMI"]
        age_pedigree  = inputs["Age"] * inputs["DiabetesPedigreeFunction"]
        raw    = np.hstack([base_vals, [[glucose_bmi, age_pedigree]]])
        imp    = imputer.transform(raw)
        orig   = imp.copy()
        scaled = scaler.transform(imp)

        prob = model.predict_proba(scaled)[0][1]
        pred = int(prob >= 0.5)
        tier = get_risk_tier(prob)

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled)

        # Display
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Tier", tier)
        c2.metric("Probability", f"{prob:.1%}")
        c3.metric("Prediction", "Diabetic" if pred == 1 else "Non-Diabetic")

        # SHAP bar
        st.markdown("#### Key Risk Drivers")
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP": shap_values[0],
            "Value": orig[0]
        }).sort_values("SHAP", key=abs, ascending=False).head(5)
        st.dataframe(shap_df.style.background_gradient(subset=["SHAP"], cmap="RdYlGn_r"), use_container_width=True)

        # Gemini
        with st.spinner("Generating Gemini recommendations ..."):
            feature_lines = "\\n".join([
                f"  - {row.Feature}: {row.Value:.1f} | SHAP={row.SHAP:+.4f}"
                for _, row in shap_df.iterrows()
            ])
            prompt = f"""You are a clinical decision support assistant.
Patient diabetes risk: {prob:.1%} ({tier}). Prediction: {"Diabetic" if pred else "Non-Diabetic"}.
Top SHAP features:\\n{feature_lines}
Return ONLY valid JSON with keys: risk_summary (string), lifestyle_advice (list of {{category, advice}}), follow_up_tests (list of {{test, urgency, reason}}).
Do not include markdown or code blocks."""
            try:
                llm = call_gemini(prompt, api_key)
                st.markdown("#### AI Risk Summary")
                st.info(llm.get("risk_summary", "N/A"))

                st.markdown("#### Lifestyle Recommendations")
                for item in llm.get("lifestyle_advice", []):
                    st.markdown(f"**{item.get('category')}:** {item.get('advice')}")

                st.markdown("#### Recommended Follow-Up Tests")
                tests = llm.get("follow_up_tests", [])
                if tests:
                    st.table(pd.DataFrame(tests))
            except Exception as e:
                st.error(f"Gemini error: {e}")
else:
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    X = X.copy()
    X["Glucose_BMI"]  = X["Glucose"].fillna(X["Glucose"].median()) * \\
                        X["BMI"].fillna(X["BMI"].median())
    X["Age_Pedigree"] = X["Age"] * X["DiabetesPedigreeFunction"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_test = y_test.reset_index(drop=True)

    patient_idx = st.slider("Select Patient (from test set)", 0, len(y_test)-1, 0)

    X_imp    = imputer.transform(X_test)
    X_orig   = X_imp.copy()
    X_scaled = scaler.transform(X_imp)

    prob = model.predict_proba(X_scaled)[patient_idx][1]
    pred = int(prob >= 0.5)
    tier = get_risk_tier(prob)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Tier", tier)
    c2.metric("Probability", f"{prob:.1%}")
    c3.metric("Prediction", "Diabetic" if pred == 1 else "Non-Diabetic")
    c4.metric("True Outcome", "Diabetic" if y_test.iloc[patient_idx] == 1 else "Non-Diabetic")

    explainer   = shap.TreeExplainer(model)
    shap_vals   = explainer.shap_values(X_scaled[patient_idx:patient_idx+1])
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP": shap_vals[0],
        "Value": X_orig[patient_idx],
        "Unit": ["count","mg/dL","mmHg","mm","mu U/mL","kg/m2","score","years","composite","composite"]
    }).sort_values("SHAP", key=abs, ascending=False).head(5)

    st.markdown("#### SHAP Risk Drivers")
    st.dataframe(shap_df.style.background_gradient(subset=["SHAP"], cmap="RdYlGn_r"), use_container_width=True)

    if st.button("🤖 Generate Gemini Recommendations", type="primary"):
        with st.spinner("Calling Gemini API ..."):
            feature_lines = "\\n".join([
                f"  - {row.Feature}: {row.Value:.1f} {row.Unit} | SHAP={row.SHAP:+.4f}"
                for _, row in shap_df.iterrows()
            ])
            prompt = f"""You are a clinical decision support assistant.
Patient risk: {prob:.1%} ({tier}). Prediction: {"Diabetic" if pred else "Non-Diabetic"}.
Top SHAP features:\\n{feature_lines}
Return ONLY valid JSON with: risk_summary, lifestyle_advice (list {{category, advice}}), follow_up_tests (list {{test, urgency, reason}}).
No markdown, no code fences."""
            try:
                llm = call_gemini(prompt, api_key)
                st.markdown("#### AI Risk Summary")
                st.info(llm.get("risk_summary"))
                st.markdown("#### Lifestyle Advice")
                for item in llm.get("lifestyle_advice", []):
                    st.success(f"**{item.get('category')}:** {item.get('advice')}")
                st.markdown("#### Follow-Up Tests")
                tests = llm.get("follow_up_tests", [])
                if tests:
                    st.table(pd.DataFrame(tests))
            except Exception as e:
                st.error(f"Gemini API error: {e}")

st.divider()
st.caption("⚠️ This tool is for clinical decision support only. All outputs must be reviewed by a qualified healthcare professional.")
'''

    with open("streamlit_app.py", "w") as f:
        f.write(app_code)
    print("[INFO] streamlit_app.py generated — run with: streamlit run streamlit_app.py")


# =============================================================================
# 9. MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "="*65)
    print("  STAGE 4: LLM-POWERED CLINICAL RECOMMENDATION ENGINE")
    print("="*65 + "\n")

    # Load model & data
    model, scaler, imputer = load_or_train_model(DATA_PATH)
    X_test_scaled, X_test_original, y_test, feature_names = prepare_test_data(
        DATA_PATH, imputer, scaler
    )

    # SHAP explainer
    print("[INFO] Computing SHAP values ...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled[:NUM_PATIENTS])

    all_reports = []

    for i in range(NUM_PATIENTS):
        print(f"\n[INFO] Processing Patient #{i+1} ...")

        prob = model.predict_proba(X_test_scaled)[i][1]
        pred = int(prob >= 0.5)
        tier, _ = get_risk_tier(prob)

        # Top 5 SHAP features
        sv = shap_values[i]
        top_idx = np.argsort(np.abs(sv))[::-1][:5]
        top_features = []
        for idx in top_idx:
            fname = feature_names[idx]
            orig_val = X_test_original[i][idx]
            top_features.append({
                "feature": fname,
                "value":   orig_val,
                "unit":    CLINICAL_RANGES.get(fname, {}).get("unit", ""),
                "status":  get_clinical_status(fname, orig_val),
                "shap":    sv[idx]
            })

        patient_data = {
            "probability":  prob,
            "predicted":    pred,
            "true_label":   int(y_test.iloc[i]),
            "risk_tier":    tier,
            "top_features": top_features
        }

        # Build prompt & call Gemini
        prompt = build_gemini_prompt(patient_data)
        print(f"  [INFO] Calling Gemini API ...")
        llm_response = call_gemini(prompt)

        # Console output
        print_patient_report(i, patient_data, llm_response)

        # Save SHAP waterfall image
        shap_img_path = os.path.join(OUTPUT_DIR, f"shap_patient_{i+1}.png")
        save_shap_waterfall(explainer, shap_values, i, feature_names, shap_img_path)

        all_reports.append({
            "patient_idx":   i,
            "patient_data":  patient_data,
            "llm_response":  llm_response,
            "shap_img_path": shap_img_path
        })

        # Rate limit — be polite to Gemini API
        time.sleep(1.5)

    # Generate PDF
    print("\n[INFO] Generating PDF report ...")
    pdf_path = os.path.join(OUTPUT_DIR, "stage4_clinical_report.pdf")
    generate_pdf_report(all_reports, pdf_path)

    # Generate Streamlit app file
    print("[INFO] Writing Streamlit app ...")
    generate_streamlit_app()

    print("\n" + "="*65)
    print("  STAGE 4 COMPLETE")
    print(f"  PDF Report  : {pdf_path}")
    print(f"  Streamlit   : streamlit run streamlit_app.py")
    print("="*65 + "\n")


if __name__ == "__main__":
    main()