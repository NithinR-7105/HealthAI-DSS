import pandas as pd
import numpy as np
import shap
import os
import warnings
warnings.filterwarnings("ignore")
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    recall_score,
    confusion_matrix
)

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

CLINICAL_RANGES = {
    "Glucose":          {"unit": "mg/dL",   "normal": (70, 99),   "prediabetes": (100, 125), "high": 126},
    "BloodPressure":    {"unit": "mmHg",    "normal": (60, 80),   "high": 90},
    "SkinThickness":    {"unit": "mm",      "normal": (10, 25)},
    "Insulin":          {"unit": "mu U/mL", "normal": (16, 166)},
    "BMI":              {"unit": "kg/m2",   "normal": (18.5, 24.9), "overweight": (25, 29.9), "obese": 30},
    "DiabetesPedigreeFunction": {"unit": "score", "note": "Genetic risk factor (higher = more risk)"},
    "Age":              {"unit": "years"},
    "Pregnancies":      {"unit": "count"},
    "Glucose_BMI":      {"unit": "composite", "note": "Interaction: high glucose × high BMI compounds risk"},
    "Age_Pedigree":     {"unit": "composite", "note": "Interaction: older age × strong family history"}
}


def get_clinical_status(feature_name, scaled_value, original_value):
    """
    Translate a feature value into a clinical status string.
    Uses original (unscaled) value if meaningful thresholds are known.
    """
    val = original_value
    if feature_name == "Glucose":
        if val >= 126:    return "HIGH (diabetic range)"
        elif val >= 100:  return "ELEVATED (pre-diabetic)"
        elif val >= 70:   return "Normal"
        else:             return "LOW"
    elif feature_name == "BMI":
        if val >= 30:     return "Obese"
        elif val >= 25:   return "Overweight"
        elif val >= 18.5: return "Normal"
        else:             return "Underweight"
    elif feature_name == "BloodPressure":
        if val >= 90:     return "HIGH"
        elif val >= 60:   return "Normal"
        else:             return "LOW"
    elif feature_name == "Age":
        if val >= 60:     return "Senior (>60)"
        elif val >= 45:   return "Middle-aged (45-60)"
        else:             return "Adult (<45)"
    elif feature_name == "DiabetesPedigreeFunction":
        if val > 0.8:     return "HIGH genetic risk"
        elif val > 0.4:   return "Moderate genetic risk"
        else:             return "Low genetic risk"
    elif feature_name == "Glucose_BMI":
        if val > 5000:    return "HIGH combined risk (elevated glucose + high BMI)"
        elif val > 3000:  return "MODERATE combined risk"
        else:             return "Lower combined risk"
    elif feature_name == "Age_Pedigree":
        if val > 30:      return "HIGH compounded risk (older age + strong family history)"
        elif val > 15:    return "MODERATE compounded risk"
        else:             return "Lower compounded risk"
    return ""

def generate_clinical_shap_explanation(
    shap_values,
    scaled_feature_values,
    original_feature_values,
    feature_names,
    prediction_prob,
    top_k=5
):
    """
    Generate a detailed, clinically-grounded SHAP explanation for a single patient.

    Parameters
    ----------
    shap_values           : 1D array of SHAP values for this patient
    scaled_feature_values : 1D array of scaled feature values (model input)
    original_feature_values : 1D array of original (unscaled) feature values
    feature_names         : list of feature name strings
    prediction_prob       : predicted probability of diabetes (0-1)
    top_k                 : number of top features to include in explanation

    Returns
    -------
    explanations : list of explanation strings
    confidence   : string describing model confidence
    risk_level   : string — "High Risk", "Moderate Risk", or "Low Risk"
    """
    abs_shap    = np.abs(shap_values)
    indices     = np.argsort(abs_shap)[::-1][:top_k]
    max_impact  = abs_shap[indices[0]]

    explanations = []

    for rank, idx in enumerate(indices, start=1):
        feature    = feature_names[idx]
        shap_val   = shap_values[idx]
        scaled_val = scaled_feature_values[idx]
        orig_val   = original_feature_values[idx]
        ratio      = abs_shap[idx] / max_impact

        direction = "INCREASED" if shap_val > 0 else "REDUCED"

        if ratio > 0.66:   impact_label = "Primary driver"
        elif ratio > 0.33: impact_label = "Secondary factor"
        else:              impact_label = "Minor contributor"

        clinical_status = get_clinical_status(feature, scaled_val, orig_val)
        unit = CLINICAL_RANGES.get(feature, {}).get("unit", "")

        status_str = f" [{clinical_status}]" if clinical_status else ""
        value_str  = f"{orig_val:.1f} {unit}".strip()

        explanations.append(
            f"  [{rank}] {feature} = {value_str}{status_str}\n"
            f"       --> {direction} diabetes risk | SHAP = {shap_val:+.4f} | {impact_label}"
        )
    if prediction_prob >= 0.80:
        confidence = "High Confidence (>= 80% probability)"
    elif prediction_prob >= 0.60:
        confidence = "Moderate Confidence (60-79% probability)"
    elif prediction_prob >= 0.50:
        confidence = "Low Confidence (50-59% probability)"
    else:
        confidence = "Likely Non-Diabetic (< 50% probability)"

    if prediction_prob >= 0.70:
        risk_level = "HIGH RISK"
    elif prediction_prob >= 0.40:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    return explanations, confidence, risk_level


def print_patient_report(patient_idx, shap_values, X_shap_scaled, X_original,
                         feature_names, y_test, model):
    """Print a full clinical risk report for a single patient."""
    pred_prob  = model.predict_proba(X_shap_scaled)[patient_idx][1]
    pred_class = model.predict(X_shap_scaled)[patient_idx]
    true_class = y_test.iloc[patient_idx]

    explanations, confidence, risk_level = generate_clinical_shap_explanation(
        shap_values=shap_values[patient_idx],
        scaled_feature_values=X_shap_scaled[patient_idx],
        original_feature_values=X_original[patient_idx],
        feature_names=feature_names,
        prediction_prob=pred_prob,
        top_k=5
    )

    outcome_str = "DIABETIC" if pred_class == 1 else "NON-DIABETIC"
    true_str    = "DIABETIC" if true_class == 1 else "NON-DIABETIC"
    correct_str = "CORRECT" if pred_class == true_class else "INCORRECT"

    print(f"\n{'='*60}")
    print(f"  PATIENT #{patient_idx + 1} CLINICAL RISK REPORT")
    print(f"{'='*60}")
    print(f"  Predicted Outcome : {outcome_str}  ({correct_str})")
    print(f"  True Outcome      : {true_str}")
    print(f"  Risk Probability  : {pred_prob:.4f}  ({risk_level})")
    print(f"  Model Confidence  : {confidence}")
    print(f"\n  Key Contributing Factors (SHAP Analysis):")
    print(f"  {'─'*55}")
    for e in explanations:
        print(e)
    print(f"{'='*60}")

DATA_PATH = "C:/Users/nithi/Desktop/diabetes_env/diabetes.csv"
df = pd.read_csv(DATA_PATH)

print(f"[INFO] Dataset: {df.shape[0]} patients, {df.shape[1]} features")
print(f"[INFO] Class distribution (0=No Diabetes, 1=Diabetes):")
print(df["Outcome"].value_counts().to_string(), "\n")

invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X = X.copy()
X["Glucose_BMI"]  = X["Glucose"].fillna(X["Glucose"].median()) * \
                    X["BMI"].fillna(X["BMI"].median())
X["Age_Pedigree"] = X["Age"] * X["DiabetesPedigreeFunction"]

print(f"[INFO] Feature engineering applied: Glucose_BMI, Age_Pedigree")
print(f"[INFO] Total features: {X.shape[1]} (excluding Outcome)\n")

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)
y_test = y_test.reset_index(drop=True)

imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp  = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)

X_test_original = X_test_imp.copy()


smote_tomek = SMOTETomek(random_state=42)
X_train_smote, y_train_smote = smote_tomek.fit_resample(X_train_scaled, y_train)
print(f"[INFO] SMOTETomek: {len(y_train)} → {len(y_train_smote)} training samples")
print(f"[INFO] Class balance after SMOTETomek: {dict(zip(*np.unique(y_train_smote, return_counts=True)))}\n")

from sklearn.model_selection import train_test_split as tts_inner
X_tr, X_val, y_tr, y_val = tts_inner(
    X_train_smote, y_train_smote, test_size=0.1, random_state=42
)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    min_child_weight=2,
    gamma=0.0,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    early_stopping_rounds=30,
    random_state=42,
    verbosity=0
)
model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val)],
    verbose=False
)
print(f"[INFO] XGBoost trained. Best iteration: {model.best_iteration}")

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy     = accuracy_score(y_test, y_pred)
bal_acc      = balanced_accuracy_score(y_test, y_pred)
sensitivity  = recall_score(y_test, y_pred)               
specificity  = recall_score(y_test, y_pred, pos_label=0)  
mae          = mean_absolute_error(y_test, y_prob)
mse          = mean_squared_error(y_test, y_prob)
roc_auc      = roc_auc_score(y_test, y_prob)

print("\n" + "="*55)
print("  FINAL HYBRID MODEL METRICS")
print("="*55)
print(f"  Accuracy          : {accuracy:.4f}")
print(f"  Balanced Accuracy : {bal_acc:.4f}   <-- robust to class imbalance")
print(f"  Sensitivity (TPR) : {sensitivity:.4f}   <-- disease detection rate")
print(f"  Specificity (TNR) : {specificity:.4f}   <-- true negative rate")
print(f"  ROC-AUC           : {roc_auc:.4f}")
print(f"  MAE               : {mae:.4f}")
print(f"  MSE               : {mse:.4f}")
print("="*55)
print("\nDetailed Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix Breakdown:")
print(f"  True Positives  (TP): {tp}  — correctly identified diabetic patients")
print(f"  True Negatives  (TN): {tn}  — correctly identified healthy patients")
print(f"  False Positives (FP): {fp}  — healthy patients flagged as diabetic (over-diagnosis)")
print(f"  False Negatives (FN): {fn}  — diabetic patients missed (critical error in medicine)")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax)
classes = ["No Diabetes", "Diabetes"]
tick_marks = np.arange(2)
ax.set_xticks(tick_marks);  ax.set_xticklabels(classes, rotation=30, ha="right")
ax.set_yticks(tick_marks);  ax.set_yticklabels(classes)
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    ax.text(j, i, cm[i, j], ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black", fontsize=14)
ax.set_ylabel("True Label");  ax.set_xlabel("Predicted Label")
ax.set_title("Confusion Matrix - Final Hybrid Model")
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/cm_stage3.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n[INFO] Initializing SHAP TreeExplainer ...")
explainer = shap.TreeExplainer(model)


X_shap_scaled   = X_test_scaled[:100]
X_shap_original = X_test_original[:100]
shap_values     = explainer.shap_values(X_shap_scaled)

print(f"[INFO] SHAP values computed for {X_shap_scaled.shape[0]} patients.")

print("\n[PLOT] Global SHAP Summary Plot (Beeswarm)...")
shap.summary_plot(
    shap_values,
    X_shap_scaled,
    feature_names=feature_names,
    show=True
)

print("\n[PLOT] Global SHAP Feature Importance (Bar)...")
shap.summary_plot(
    shap_values,
    X_shap_scaled,
    feature_names=feature_names,
    plot_type="bar",
    show=True
)


shap_importance_mean = np.abs(shap_values).mean(axis=0)
top2_indices = np.argsort(shap_importance_mean)[::-1][:2]

print("\n[PLOT] SHAP Dependence Plots for top 2 features...")
for idx in top2_indices:
    shap.dependence_plot(
        idx,
        shap_values,
        X_shap_scaled,
        feature_names=feature_names,
        show=True
    )


y_pred_100 = model.predict(X_shap_scaled)
y_test_100 = y_test.iloc[:100].reset_index(drop=True)


case_indices = {}
for i in range(len(y_test_100)):
    true = y_test_100.iloc[i]
    pred = y_pred_100[i]
    if true == 1 and pred == 1 and "TP" not in case_indices:
        case_indices["TP"] = i
    elif true == 0 and pred == 0 and "TN" not in case_indices:
        case_indices["TN"] = i
    elif true == 1 and pred == 0 and "FN" not in case_indices:
        case_indices["FN"] = i
    elif true == 0 and pred == 1 and "FP" not in case_indices:
        case_indices["FP"] = i
    if len(case_indices) == 4:
        break

case_labels = {"TP": "TRUE POSITIVE", "TN": "TRUE NEGATIVE",
               "FN": "FALSE NEGATIVE (Missed Case)", "FP": "FALSE POSITIVE (Over-diagnosis)"}

print("\n" + "="*60)
print("  SECTION 12: CLINICAL CASE INTERPRETATION (SHAP)")
print("="*60)

for case_type, patient_idx in case_indices.items():
    print(f"\n  --- Case Type: {case_labels[case_type]} ---")
    print_patient_report(
        patient_idx=patient_idx,
        shap_values=shap_values,
        X_shap_scaled=X_shap_scaled,
        X_original=X_shap_original,
        feature_names=feature_names,
        y_test=y_test_100,
        model=model
    )

    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[patient_idx],
        feature_names=feature_names,
        max_display=8,
        show=True
    )


print("\n[PLOT] Cohort-level SHAP comparison (Diabetic vs Non-Diabetic)...")

diabetic_idx     = np.where(y_test_100 == 1)[0]
nondiabetic_idx  = np.where(y_test_100 == 0)[0]

if len(diabetic_idx) > 0 and len(nondiabetic_idx) > 0:
    mean_shap_diabetic    = np.abs(shap_values[diabetic_idx]).mean(axis=0)
    mean_shap_nondiabetic = np.abs(shap_values[nondiabetic_idx]).mean(axis=0)

    shap_df = pd.DataFrame({
        "Feature":       feature_names,
        "Diabetic":      mean_shap_diabetic,
        "Non-Diabetic":  mean_shap_nondiabetic
    }).sort_values("Diabetic", ascending=False)

    x_pos = np.arange(len(feature_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, shap_df["Diabetic"],
           width, label="Diabetic Group",    color="tomato",    alpha=0.85)
    ax.bar(x_pos + width/2, shap_df["Non-Diabetic"],
           width, label="Non-Diabetic Group", color="steelblue", alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(shap_df["Feature"], rotation=25, ha="right")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.set_title("Cohort SHAP Comparison: Diabetic vs Non-Diabetic Patients")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/cohort_shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n  Top risk drivers for DIABETIC patients:")
    for _, row in shap_df.head(3).iterrows():
        print(f"    {row['Feature']:<30} SHAP importance: {row['Diabetic']:.4f}")


shap_importance = np.abs(shap_values).mean(axis=0)
shap_df_global = pd.DataFrame({
    "Feature":    feature_names,
    "Importance": shap_importance
}).sort_values("Importance", ascending=True) 

fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(shap_df_global)))
ax.barh(shap_df_global["Feature"], shap_df_global["Importance"], color=colors)
ax.set_xlabel("Mean |SHAP Value| (Feature Impact on Prediction)")
ax.set_title("Key Clinical Factors Influencing Diabetes Risk\n(SHAP Global Importance)")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("results/shap_global_importance.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "="*60)
print("  PROJECT CONTRIBUTIONS")
print("="*60)
print("""
  1. HYBRID AUGMENTATION PIPELINE
     Combined SMOTE oversampling with XGBoost to address class
     imbalance, improving minority-class (diabetic) detection.

  2. CLINICALLY MEANINGFUL EVALUATION
     Reported Sensitivity and Specificity alongside accuracy —
     essential metrics for medical diagnostic systems.

  3. EXPLAINABLE AI (SHAP) FOR CLINICAL TRUST
     SHAP-based explanations provide transparent, feature-level
     reasoning for each individual prediction — enabling
     clinicians to verify and audit model decisions.

  4. CLINICAL CASE INTERPRETATION
     Detailed per-patient reports mapping SHAP values to
     clinical context (thresholds, units, risk categories).

  5. COHORT-LEVEL INSIGHTS
     Group-level SHAP analysis reveals which features drive
     risk differently across diabetic and non-diabetic cohorts.
""")

print("="*60)
print("  LIMITATIONS")
print("="*60)
print("""
  1. DATASET SCOPE
     Pima Indians Diabetes Dataset is limited to female patients
     of a single ethnic group; generalizability is restricted.

  2. SYNTHETIC OVERSAMPLING
     SMOTE generates interpolated samples in feature space, not
     real clinical data — synthetic examples may not reflect
     true physiological variability.

  3. FEATURE AVAILABILITY
     Insulin and SkinThickness have high missing rates (~49%,
     ~30%), filled by median imputation which can reduce variance.

  4. STATIC PREDICTIONS
     The model produces a snapshot prediction; no temporal
     or longitudinal patient data is incorporated.

  5. SHAP APPROXIMATION
     SHAP values for tree models are exact but represent
     marginal contributions under a specific background
     distribution — interpretation should be domain-validated.
""")

print("[DONE] Stage 3 complete. All results saved to 'results/' folder.")

os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(imputer, "model/imputer.pkl")

print("Model saved successfully.")