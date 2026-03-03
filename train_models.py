"""
COVID-19 Mortality Prediction — Full Training Pipeline
======================================================
Runs EDA, trains 5 models (Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, MLP), performs SHAP analysis, and saves all artifacts to disk.

Usage:  python3 train_models.py
"""

import os, json, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             classification_report, confusion_matrix)
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------- paths ----------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS  = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
METRICS_DIR = os.path.join(BASE, "metrics")
DATA   = os.path.join(BASE, "data")
for d in [PLOTS, MODELS, METRICS_DIR, DATA]:
    os.makedirs(d, exist_ok=True)

# ---------- 1. Load Data ---------------------------------------------------
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
df = pd.read_csv(os.path.join(BASE, "covid_cleaned3.csv"))
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic stats:\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Save descriptive stats
stats = {
    "n_rows": int(df.shape[0]),
    "n_features": int(df.shape[1] - 1),
    "feature_names": list(df.columns.drop("DEATH")),
    "target": "DEATH",
    "death_counts": df["DEATH"].value_counts().to_dict(),
    "numerical_features": ["AGE"],
    "categorical_features": [c for c in df.columns if c not in ["AGE", "DEATH"]],
}

# ================== PART 1: DESCRIPTIVE ANALYTICS =========================
print("\n" + "=" * 60)
print("PART 1: DESCRIPTIVE ANALYTICS")
print("=" * 60)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight",
                      "font.size": 11, "axes.titlesize": 13})

# --- 1.2 Target Distribution ---
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["DEATH"].value_counts()
bars = ax.bar(["Survived (0)", "Died (1)"], counts.values,
              color=["#2ecc71", "#e74c3c"], edgecolor="black", width=0.5)
for bar, v in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 5000,
            f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Distribution of Target Variable (DEATH)")
ax.set_ylabel("Count")
ax.set_xlabel("Outcome")
fig.savefig(os.path.join(PLOTS, "target_distribution.png"))
plt.close(fig)
print("✓ Saved target_distribution.png")

# --- 1.3a Age Distribution by Mortality (Violin Plot) ---
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df, x="DEATH", y="AGE", palette=["#2ecc71", "#e74c3c"],
               inner="quartile", ax=ax)
ax.set_xticklabels(["Survived", "Died"])
ax.set_title("Age Distribution by Mortality Outcome")
ax.set_ylabel("Age")
ax.set_xlabel("Outcome")
fig.savefig(os.path.join(PLOTS, "age_violin.png"))
plt.close(fig)
print("✓ Saved age_violin.png")

# --- 1.3b Comorbidity Prevalence by Mortality ---
comorbidities = ["DIABETES", "COPD", "ASTHMA", "HYPERTENSION",
                 "OBESITY", "CARDIOVASCULAR", "RENAL_CHRONIC",
                 "IMMUNOSUPPRESSION", "TOBACCO"]
mort_rates = []
for c in comorbidities:
    subset = df[df[c] == 1]
    rate = subset["DEATH"].mean() * 100 if len(subset) > 0 else 0
    mort_rates.append(rate)

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(comorbidities, mort_rates, color=sns.color_palette("Reds_r", len(comorbidities)),
               edgecolor="black")
for bar, v in zip(bars, mort_rates):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{v:.1f}%", va="center", fontweight="bold")
ax.set_title("Mortality Rate by Comorbidity")
ax.set_xlabel("Mortality Rate (%)")
fig.savefig(os.path.join(PLOTS, "comorbidity_mortality.png"))
plt.close(fig)
print("✓ Saved comorbidity_mortality.png")

# --- 1.3c Hospitalization & Pneumonia vs Mortality (Grouped Bar) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, feat in zip(axes, ["HOSPITALIZED", "PNEUMONIA"]):
    ct = pd.crosstab(df[feat], df["DEATH"], normalize="index") * 100
    ct.plot(kind="bar", stacked=True, ax=ax,
            color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_title(f"Mortality Rate by {feat}")
    ax.set_xlabel(feat)
    ax.set_ylabel("Percentage")
    ax.set_xticklabels(["No", "Yes"], rotation=0)
    ax.legend(["Survived", "Died"], loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "hospitalized_pneumonia_mortality.png"))
plt.close(fig)
print("✓ Saved hospitalized_pneumonia_mortality.png")

# --- 1.3d Age Histogram by Mortality ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df[df["DEATH"] == 0]["AGE"], bins=50, alpha=0.6, label="Survived",
        color="#2ecc71", edgecolor="black")
ax.hist(df[df["DEATH"] == 1]["AGE"], bins=50, alpha=0.6, label="Died",
        color="#e74c3c", edgecolor="black")
ax.set_title("Age Distribution: Survived vs. Died")
ax.set_xlabel("Age")
ax.set_ylabel("Count")
ax.legend()
fig.savefig(os.path.join(PLOTS, "age_histogram.png"))
plt.close(fig)
print("✓ Saved age_histogram.png")

# --- 1.3e COVID Positive vs Mortality ---
fig, ax = plt.subplots(figsize=(7, 5))
ct = pd.crosstab(df["COVID_POSITIVE"], df["DEATH"], normalize="index") * 100
ct.plot(kind="bar", stacked=True, ax=ax,
        color=["#2ecc71", "#e74c3c"], edgecolor="black")
ax.set_title("Mortality Rate by COVID Test Result")
ax.set_xlabel("COVID Positive")
ax.set_ylabel("Percentage")
ax.set_xticklabels(["Negative/Pending", "Positive"], rotation=0)
ax.legend(["Survived", "Died"], loc="upper right")
fig.savefig(os.path.join(PLOTS, "covid_positive_mortality.png"))
plt.close(fig)
print("✓ Saved covid_positive_mortality.png")

# --- 1.4 Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix Heatmap")
fig.savefig(os.path.join(PLOTS, "correlation_heatmap.png"))
plt.close(fig)
print("✓ Saved correlation_heatmap.png")

# ================== PART 2: PREDICTIVE ANALYTICS ==========================
print("\n" + "=" * 60)
print("PART 2: PREDICTIVE ANALYTICS")
print("=" * 60)

# --- 2.1 Data Preparation ---
# Balance: take ALL deaths, randomly match with equal number of survivors
df_died = df[df["DEATH"] == 1]
n_died = len(df_died)
print(f"\nBalancing dataset: all {n_died:,} deaths + {n_died:,} random survivors...")
df_survived = df[df["DEATH"] == 0].sample(n=n_died, random_state=42)
df_balanced = pd.concat([df_died, df_survived]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Balanced dataset shape: {df_balanced.shape}")
print(f"Class distribution:\n{df_balanced['DEATH'].value_counts()}")

X = df_balanced.drop("DEATH", axis=1)
y = df_balanced["DEATH"]
feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

# Scale AGE only
scaler = StandardScaler()
age_idx = feature_names.index("AGE")
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled["AGE"] = scaler.fit_transform(X_train[["AGE"]])
X_test_scaled["AGE"] = scaler.transform(X_test[["AGE"]])

joblib.dump(scaler, os.path.join(DATA, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(DATA, "feature_names.pkl"))
print("✓ Scaler and feature names saved")

all_metrics = {}

def evaluate_model(name, model, X_te, y_te, y_prob=None):
    """Compute classification metrics and store them."""
    y_pred = model.predict(X_te) if hasattr(model, 'predict') else None
    if y_pred is None:
        return {}
    if y_prob is None:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_te)
    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, y_prob) if y_prob is not None else 0
    m = {"Accuracy": round(acc, 4), "Precision": round(prec, 4),
         "Recall": round(rec, 4), "F1": round(f1, 4), "AUC-ROC": round(auc, 4)}
    all_metrics[name] = m
    print(f"\n  {name} Metrics:")
    for k, v in m.items():
        print(f"    {k}: {v}")
    return m

# --- 2.2 Logistic Regression Baseline ---
print("\n--- 2.2 Logistic Regression ---")
lr = LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
evaluate_model("Logistic Regression", lr, X_test_scaled, y_test, lr_prob)
joblib.dump(lr, os.path.join(MODELS, "logistic_regression.pkl"))
print("✓ Logistic Regression saved")

# ROC for LR
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)

# --- 2.3 Decision Tree ---
print("\n--- 2.3 Decision Tree ---")
dt_params = {"max_depth": [3, 5, 7, 10],
             "min_samples_leaf": [5, 10, 20, 50]}
dt_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                     dt_params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
dt_cv.fit(X_train, y_train)
dt_best = dt_cv.best_estimator_
print(f"  Best params: {dt_cv.best_params_}")
dt_prob = dt_best.predict_proba(X_test)[:, 1]
evaluate_model("Decision Tree", dt_best, X_test, y_test, dt_prob)
joblib.dump(dt_best, os.path.join(MODELS, "decision_tree.pkl"))
all_metrics["Decision Tree"]["best_params"] = dt_cv.best_params_
print("✓ Decision Tree saved")

fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_prob)

# Visualize best tree
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(dt_best, feature_names=feature_names, class_names=["Survived", "Died"],
          filled=True, rounded=True, ax=ax, max_depth=3, fontsize=9)
ax.set_title(f"Decision Tree (max_depth={dt_cv.best_params_['max_depth']})")
fig.savefig(os.path.join(PLOTS, "decision_tree_viz.png"))
plt.close(fig)
print("✓ Saved decision_tree_viz.png")

# --- 2.4 Random Forest ---
print("\n--- 2.4 Random Forest ---")
rf_params = {"n_estimators": [50, 100, 200],
             "max_depth": [3, 5, 8]}
rf_cv = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
                     rf_params, cv=5, scoring="f1", n_jobs=-1, verbose=1)
rf_cv.fit(X_train, y_train)
rf_best = rf_cv.best_estimator_
print(f"  Best params: {rf_cv.best_params_}")
rf_prob = rf_best.predict_proba(X_test)[:, 1]
evaluate_model("Random Forest", rf_best, X_test, y_test, rf_prob)
joblib.dump(rf_best, os.path.join(MODELS, "random_forest.pkl"))
all_metrics["Random Forest"]["best_params"] = rf_cv.best_params_
print("✓ Random Forest saved")

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)

# --- 2.5 Gradient Boosting ---
print("\n--- 2.5 Gradient Boosting ---")
gb_params = {"n_estimators": [50, 100, 200],
             "max_depth": [3, 4, 5],
             "learning_rate": [0.01, 0.05, 0.1]}
gb_cv = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params, cv=3, scoring="f1", n_jobs=-1, verbose=1)
gb_cv.fit(X_train, y_train)
gb_best = gb_cv.best_estimator_
print(f"  Best params: {gb_cv.best_params_}")
gb_prob = gb_best.predict_proba(X_test)[:, 1]
evaluate_model("Gradient Boosting", gb_best, X_test, y_test, gb_prob)
joblib.dump(gb_best, os.path.join(MODELS, "gradient_boosting.pkl"))
all_metrics["Gradient Boosting"]["best_params"] = gb_cv.best_params_
print("✓ Gradient Boosting saved")

fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_prob)

# --- 2.6 Neural Network (MLP w/ sklearn) ---
print("\n--- 2.6 Neural Network (MLP) ---")
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=200,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    verbose=True
)
mlp.fit(X_train_scaled, y_train)

# Plot training history (loss curve)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mlp.loss_curve_, label="Train Loss")
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ is not None:
    ax.plot(mlp.validation_scores_, label="Validation Accuracy")
ax.set_title("MLP Training History")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss / Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "mlp_training_history.png"))
plt.close(fig)
print("✓ Saved mlp_training_history.png")

# Evaluate MLP
nn_prob = mlp.predict_proba(X_test_scaled)[:, 1]
evaluate_model("Neural Network (MLP)", mlp, X_test_scaled, y_test, nn_prob)
joblib.dump(mlp, os.path.join(MODELS, "mlp_model.pkl"))
print("✓ MLP model saved")

fpr_nn, tpr_nn, _ = roc_curve(y_test, nn_prob)

# --- 2.7 Model Comparison ---
print("\n--- 2.7 Model Comparison ---")
comp_df = pd.DataFrame(all_metrics).T
# Drop best_params for the display table
display_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
comp_display = comp_df[display_cols]
print("\n" + comp_display.to_string())
comp_display.to_csv(os.path.join(METRICS_DIR, "model_comparison.csv"))

# Bar chart of F1 scores
fig, ax = plt.subplots(figsize=(10, 6))
models_list = comp_display.index.tolist()
f1_scores = comp_display["F1"].values.astype(float)
colors = sns.color_palette("viridis", len(models_list))
bars = ax.bar(models_list, f1_scores, color=colors, edgecolor="black")
for bar, v in zip(bars, f1_scores):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
            f"{v:.4f}", ha="center", fontweight="bold", fontsize=10)
ax.set_title("Model Comparison — F1 Score")
ax.set_ylabel("F1 Score")
ax.set_ylim(0, max(f1_scores) * 1.15)
plt.xticks(rotation=15, ha="right")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "model_comparison_f1.png"))
plt.close(fig)
print("✓ Saved model_comparison_f1.png")

# Combined ROC Curve
fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(fpr_lr, tpr_lr, label=f"Logistic Reg (AUC={all_metrics['Logistic Regression']['AUC-ROC']:.4f})")
ax.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC={all_metrics['Decision Tree']['AUC-ROC']:.4f})")
ax.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={all_metrics['Random Forest']['AUC-ROC']:.4f})")
ax.plot(fpr_gb, tpr_gb, label=f"Gradient Boosting (AUC={all_metrics['Gradient Boosting']['AUC-ROC']:.4f})")
ax.plot(fpr_nn, tpr_nn, label=f"MLP (AUC={all_metrics['Neural Network (MLP)']['AUC-ROC']:.4f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
ax.set_title("ROC Curves — All Models")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
fig.savefig(os.path.join(PLOTS, "roc_curves_all.png"))
plt.close(fig)
print("✓ Saved roc_curves_all.png")

# ================== PART 3: SHAP EXPLAINABILITY ===========================
print("\n" + "=" * 60)
print("PART 3: SHAP ANALYSIS")
print("=" * 60)

# Use Gradient Boosting for SHAP (best performing tree model)
# Use a subsample for speed
print("Computing SHAP values (subsample for speed)...")
shap_n = min(2000, len(X_test))
X_shap = X_test.sample(n=shap_n, random_state=42)

explainer = shap.TreeExplainer(gb_best)
shap_values = explainer(X_shap)

# SHAP Summary (Beeswarm)
fig = plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_shap, feature_names=feature_names, show=False)
plt.title("SHAP Summary (Beeswarm) — Gradient Boosting")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, "shap_summary_beeswarm.png"))
plt.close(fig)
print("✓ Saved shap_summary_beeswarm.png")

# SHAP Bar Plot
fig = plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, show=False)
plt.title("Mean |SHAP| Feature Importance — Gradient Boosting")
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, "shap_bar.png"))
plt.close(fig)
print("✓ Saved shap_bar.png")

# SHAP Waterfall for one prediction (high-risk individual)
# Find a patient who died
died_indices = X_shap.index[y_test.loc[X_shap.index] == 1]
if len(died_indices) > 0:
    idx = 0  # first died patient in subsample
    fig = plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_values[idx], show=False)
    plt.title("SHAP Waterfall — High-Risk Individual")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, "shap_waterfall.png"))
    plt.close(fig)
    print("✓ Saved shap_waterfall.png")

# Save SHAP explainer for Streamlit
joblib.dump(explainer, os.path.join(MODELS, "shap_explainer.pkl"))

# ================== SAVE ALL METRICS ======================================
# Serialize safely — convert numpy types
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

save_data = {
    "metrics": make_serializable(all_metrics),
    "stats": make_serializable(stats),
    "feature_names": feature_names,
}
with open(os.path.join(METRICS_DIR, "metrics.json"), "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\n" + "=" * 60)
print("ALL DONE! Artifacts saved to:")
print(f"  Models:  {MODELS}")
print(f"  Plots:   {PLOTS}")
print(f"  Metrics: {METRICS_DIR}")
print(f"  Data:    {DATA}")
print("=" * 60)
