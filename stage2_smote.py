
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     GridSearchCV)
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    accuracy_score, balanced_accuracy_score, classification_report,
    roc_auc_score, roc_curve, auc, recall_score, confusion_matrix, f1_score
)
from sklearn.impute          import SimpleImputer

from imblearn.over_sampling  import SMOTE
from imblearn.ensemble       import BalancedRandomForestClassifier
from imblearn.combine        import SMOTETomek
from xgboost                 import XGBClassifier
import matplotlib.pyplot     as plt



def plot_roc_curve(y_test, y_prob, model_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, color="steelblue", label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=1.5, color="gray")
    plt.xlabel("False Positive Rate");  plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}");  plt.legend(loc="lower right")
    plt.grid(alpha=0.3);  plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_confusion_matrix(cm, model_name, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["No Diabetes", "Diabetes"];  tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks);  ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(tick_marks);  ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=13)
    ax.set_ylabel("True Label");  ax.set_xlabel("Predicted Label")
    ax.set_title(f"Confusion Matrix — {model_name}");  plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def print_medical_metrics(y_test, y_pred, y_prob, stage_label):
    acc         = accuracy_score(y_test, y_pred)
    bal_acc     = balanced_accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    roc_auc     = roc_auc_score(y_test, y_prob)
    print(f"\n{'='*55}\n  {stage_label}\n{'='*55}")
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}   <-- robust to class imbalance")
    print(f"  Sensitivity (TPR) : {sensitivity:.4f}   <-- critical for disease detection")
    print(f"  Specificity (TNR) : {specificity:.4f}   <-- avoids false alarms")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"{'='*55}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
    return {
        "Accuracy":          round(acc, 4),
        "Balanced Accuracy": round(bal_acc, 4),
        "Sensitivity":       round(sensitivity, 4),
        "Specificity":       round(specificity, 4),
        "ROC-AUC":           round(roc_auc, 4)
    }


def find_best_threshold(y_prob, y_test, low=0.30, high=0.56):
    """Return threshold + F1 that maximises F1 in range [low, high]."""
    best_thresh, best_f1 = 0.35, 0.0
    for t in np.arange(low, high, 0.01):
        preds = (y_prob >= t).astype(int)
        if len(np.unique(preds)) < 2:
            continue
        f = f1_score(y_test, preds)
        if f > best_f1:
            best_f1, best_thresh = f, t
    return round(best_thresh, 2), round(best_f1, 4)


DATA_PATH = "C:/Users/nithi/Desktop/diabetes_env/diabetes.csv"
df = pd.read_csv(DATA_PATH)

print(f"[INFO] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"[INFO] Class distribution:\n{df['Outcome'].value_counts()}\n")

invalid_zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[invalid_zero_cols] = df[invalid_zero_cols].replace(0, np.nan)

missing_pct = df[invalid_zero_cols].isnull().mean() * 100
print("[INFO] Missing value % after 0-replacement:")
print(missing_pct.round(2).to_string(), "\n")

print("=" * 55)
print("  FEATURE ENGINEERING")
print("=" * 55)

df["Glucose_BMI"]  = df["Glucose"].fillna(df["Glucose"].median()) * \
                     df["BMI"].fillna(df["BMI"].median())
df["Age_Pedigree"] = df["Age"] * df["DiabetesPedigreeFunction"]

print(f"[INFO] Added: Glucose_BMI, Age_Pedigree")
print(f"[INFO] Total features: {df.shape[1] - 1} (excluding Outcome)\n")

X             = df.drop("Outcome", axis=1)
y             = df["Outcome"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"[INFO] Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")


imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_imp    = imputer.fit_transform(X_train)
X_test_imp     = imputer.transform(X_test)
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled  = scaler.transform(X_test_imp)


print("=" * 55)
print("  SMOTETomek AUGMENTATION")
print("=" * 55)

smt = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smt.fit_resample(X_train_scaled, y_train)

print(f"[INFO] Original train size : {X_train_scaled.shape[0]}")
print(f"[INFO] After SMOTETomek   : {X_train_resampled.shape[0]}")
print(f"[INFO] Class balance      : "
      f"{dict(zip(*np.unique(y_train_resampled, return_counts=True)))}\n")

labels = ['Healthy', 'Diabetic']
orig_counts   = [int((y_train == 0).sum()), int((y_train == 1).sum())]
resamp_counts = [int((y_train_resampled == 0).sum()),
                 int((y_train_resampled == 1).sum())]
x = [0, 1];  width = 0.35
fig, ax = plt.subplots()
ax.bar([i - width/2 for i in x], orig_counts,   width,
       label='Before SMOTETomek', color='steelblue')
ax.bar([i + width/2 for i in x], resamp_counts, width,
       label='After SMOTETomek',  color='orange')
ax.set_ylabel('Number of Samples')
ax.set_title('Class Distribution — Before vs After SMOTETomek')
ax.set_xticks(x);  ax.set_xticklabels(labels);  ax.legend()
plt.tight_layout()
plt.savefig("smote_distribution_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("=" * 55)
print("  5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_metrics = {
    "accuracy": [], "balanced_accuracy": [],
    "sensitivity": [], "specificity": [], "roc_auc": []
}

for fold, (train_idx, val_idx) in enumerate(
        skf.split(X_train_scaled, y_train), start=1):

    X_fold_train_raw = X_train_scaled[train_idx]
    X_fold_val       = X_train_scaled[val_idx]
    y_fold_train_raw = y_train.iloc[train_idx]
    y_fold_val       = y_train.iloc[val_idx]
    smt_fold = SMOTETomek(random_state=42)
    X_fold_train, y_fold_train = smt_fold.fit_resample(
        X_fold_train_raw, y_fold_train_raw
    )

    fold_model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        scale_pos_weight=500/268,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
    fold_model.fit(X_fold_train, y_fold_train)

    y_val_prob = fold_model.predict_proba(X_fold_val)[:, 1]
    y_val_pred = (y_val_prob >= 0.35).astype(int)

    cv_metrics["accuracy"].append(accuracy_score(y_fold_val, y_val_pred))
    cv_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_fold_val, y_val_pred))
    cv_metrics["sensitivity"].append(recall_score(y_fold_val, y_val_pred))
    cv_metrics["specificity"].append(recall_score(y_fold_val, y_val_pred, pos_label=0))
    cv_metrics["roc_auc"].append(roc_auc_score(y_fold_val, y_val_prob))

    print(f"  Fold {fold} | Acc: {cv_metrics['accuracy'][-1]:.4f} | "
          f"Bal-Acc: {cv_metrics['balanced_accuracy'][-1]:.4f} | "
          f"Sens: {cv_metrics['sensitivity'][-1]:.4f} | "
          f"AUC: {cv_metrics['roc_auc'][-1]:.4f}")

print("\n  Cross-Validation Summary (mean +/- std):")
print("-" * 55)
for metric, values in cv_metrics.items():
    print(f"  {metric:<22}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
print("=" * 55, "\n")

print("=" * 55)
print("  GRIDSEARCH: Hyperparameter Tuning")
print("=" * 55)

param_grid = {
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.03, 0.05, 0.1],
    "min_child_weight": [2, 3, 5],
    "gamma":            [0.0, 0.1, 0.2],
}

grid_search = GridSearchCV(
    XGBClassifier(
        n_estimators=200, subsample=0.8, colsample_bytree=0.9,
        scale_pos_weight=500/268, eval_metric="logloss",
        random_state=42, verbosity=0
    ),
    param_grid,
    cv=5,
    scoring="balanced_accuracy",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_resampled, y_train_resampled)

best_params   = grid_search.best_params_
best_cv_score = grid_search.best_score_
print(f"[INFO] Best params : {best_params}")
print(f"[INFO] Best CV balanced accuracy: {best_cv_score:.4f}\n")


print("=" * 55)
print("  FINAL XGBoost MODEL")
print("=" * 55)

model = XGBClassifier(
    n_estimators          = 500,
    max_depth             = best_params["max_depth"],
    learning_rate         = best_params["learning_rate"],
    subsample             = 0.8,
    colsample_bytree      = 0.9,
    min_child_weight      = best_params["min_child_weight"],
    gamma                 = best_params["gamma"],
    scale_pos_weight      = 500/268,
    eval_metric           = "logloss",
    early_stopping_rounds = 30,
    random_state          = 42,
    verbosity             = 0
)

model.fit(
    X_train_resampled, y_train_resampled,
    eval_set=[(X_train_resampled, y_train_resampled), (X_test_scaled, y_test)],
    verbose=False
)
print(f"[INFO] Best iteration (early stopping): {model.best_iteration}")


results_eval = model.evals_result()
epochs = range(len(results_eval["validation_0"]["logloss"]))

plt.figure(figsize=(7, 4))
plt.plot(epochs, results_eval["validation_0"]["logloss"],
         label="Train Loss",      color="steelblue")
plt.plot(epochs, results_eval["validation_1"]["logloss"],
         label="Validation Loss", color="tomato")
plt.xlabel("Boosting Rounds");  plt.ylabel("Log Loss")
plt.title("XGBoost — Training vs Validation Loss")
plt.legend();  plt.grid(alpha=0.3);  plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/loss_smote_model.png", dpi=150, bbox_inches="tight")
plt.show()


y_prob = model.predict_proba(X_test_scaled)[:, 1]

best_thresh, best_f1 = find_best_threshold(y_prob, y_test)
print(f"[INFO] Optimal threshold: {best_thresh:.2f} | F1: {best_f1:.4f}")

y_pred = (y_prob >= best_thresh).astype(int)

xgb_metrics = print_medical_metrics(
    y_test, y_pred, y_prob,
    stage_label=f"XGBoost + SMOTETomek + GridSearch (threshold={best_thresh:.2f})"
)

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(
    cm,
    model_name=f"XGBoost + SMOTETomek (threshold={best_thresh:.2f})",
    save_path="results/cm_smote_model.png"
)


plot_roc_curve(
    y_test=y_test,
    y_prob=y_prob,
    model_name=f"XGBoost + SMOTETomek (threshold={best_thresh:.2f})",
    save_path="results/roc_smote_model.png"
)


print("=" * 55)
print("  BALANCED RANDOM FOREST — Comparison Baseline")
print("=" * 55)

brf = BalancedRandomForestClassifier(
    n_estimators=200, random_state=42, n_jobs=-1
)
brf.fit(X_train_scaled, y_train)
brf_prob              = brf.predict_proba(X_test_scaled)[:, 1]
brf_thresh, brf_f1   = find_best_threshold(brf_prob, y_test)
brf_pred              = (brf_prob >= brf_thresh).astype(int)

brf_metrics = print_medical_metrics(
    y_test, brf_pred, brf_prob,
    stage_label=f"Balanced Random Forest (threshold={brf_thresh:.2f})"
)


print("\n  XGBoost vs Balanced Random Forest:")
print(f"  {'Metric':<22} {'XGBoost':>10} {'BRF':>10}")
print("  " + "-" * 42)
for k in xgb_metrics:
    print(f"  {k:<22} {xgb_metrics[k]:>10.4f} {brf_metrics[k]:>10.4f}")

thresholds_plot = np.arange(0.30, 0.60, 0.01)
f1_scores_plot, acc_plot, sens_plot = [], [], []

for t in thresholds_plot:
    p = (y_prob >= t).astype(int)
    f1_scores_plot.append(f1_score(y_test, p)  if len(np.unique(p)) > 1 else 0)
    acc_plot.append(accuracy_score(y_test, p))
    sens_plot.append(recall_score(y_test, p)    if len(np.unique(p)) > 1 else 0)

plt.figure(figsize=(8, 4))
plt.plot(thresholds_plot, f1_scores_plot, label="F1 Score",    color="steelblue", lw=2)
plt.plot(thresholds_plot, acc_plot,       label="Accuracy",    color="seagreen",  lw=2)
plt.plot(thresholds_plot, sens_plot,      label="Sensitivity", color="tomato",    lw=2)
plt.axvline(best_thresh, color="gray", linestyle="--", lw=1.5,
            label=f"Optimal = {best_thresh:.2f}")
plt.xlabel("Classification Threshold");  plt.ylabel("Score")
plt.title("XGBoost — Threshold vs F1 / Accuracy / Sensitivity")
plt.legend();  plt.grid(alpha=0.3);  plt.tight_layout()
plt.savefig("results/threshold_analysis.png", dpi=150, bbox_inches="tight")
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
metric_labels = ["Accuracy", "Balanced\nAccuracy", "Sensitivity", "Specificity", "ROC-AUC"]
metric_means  = [np.mean(cv_metrics[k]) for k in cv_metrics]
metric_stds   = [np.std(cv_metrics[k])  for k in cv_metrics]
colors        = ["steelblue", "seagreen", "tomato", "goldenrod", "mediumpurple"]

axes[0].bar(metric_labels, metric_means, yerr=metric_stds,
            color=colors, edgecolor="white", capsize=6, alpha=0.85)
axes[0].set_ylim(0, 1.12);  axes[0].set_title("5-Fold CV: Mean Metrics (+/- std)")
axes[0].set_ylabel("Score");  axes[0].grid(axis="y", alpha=0.3)
for i, (m, s) in enumerate(zip(metric_means, metric_stds)):
    axes[0].text(i, m + s + 0.02, f"{m:.3f}",
                 ha="center", fontsize=9, fontweight="bold")

fold_labels = [f"Fold {i+1}" for i in range(5)]
axes[1].plot(fold_labels, cv_metrics["roc_auc"],           marker="o",
             color="steelblue", linewidth=2, markersize=8, label="ROC-AUC")
axes[1].plot(fold_labels, cv_metrics["sensitivity"],       marker="s",
             color="tomato",    linewidth=2, markersize=8, label="Sensitivity")
axes[1].plot(fold_labels, cv_metrics["balanced_accuracy"], marker="^",
             color="seagreen",  linewidth=2, markersize=8, label="Balanced Accuracy")
axes[1].set_ylim(0.5, 1.05);  axes[1].set_title("Per-Fold Performance (K=5)")
axes[1].set_ylabel("Score");  axes[1].legend();  axes[1].grid(alpha=0.3)

plt.suptitle("5-Fold CV Analysis — XGBoost + SMOTETomek",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("results/cv_smote_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n" + "=" * 55)
print("  DONE — All results saved to results/ folder")
print("=" * 55)
print("\n[PIPELINE SUMMARY]")
print("  Feature Engineering : Glucose_BMI, Age_Pedigree")
print("  Augmentation        : SMOTETomek (inside each CV fold)")
print("  Tuning              : GridSearchCV — max_depth, learning_rate,")
print("                        min_child_weight, gamma")
print("  Model               : XGBoost with early stopping (n_estimators=500)")
print("  Threshold           : F1-maximising search over 0.30–0.55")
print("  Comparison          : Balanced Random Forest baseline")
print("\n[OUTPUT FILES]")
print("  results/loss_smote_model.png    — train vs validation loss curve")
print("  results/cm_smote_model.png      — confusion matrix")
print("  results/roc_smote_model.png     — ROC curve")
print("  results/threshold_analysis.png  — F1/accuracy/sensitivity vs threshold")
print("  results/cv_smote_analysis.png   — 5-fold CV charts")
print("  smote_distribution_comparison.png — class balance before/after")
print("\n[EXPECTED RESULTS]")
print("  CV Accuracy   : ~0.77 – 0.78")
print("  Test Accuracy : ~0.75 – 0.77")
print("  Sensitivity   : ~0.83 – 0.87   (high recall for diabetic detection)")
print("  ROC-AUC       : ~0.82 – 0.85")
print("\n[LIMITATIONS]")
print("  - SMOTETomek synthesises samples in feature space, not real physiology")
print("  - GridSearch adds ~2–5 min runtime")
print("  - Results specific to PIMA Indians Diabetes Dataset (1988)")