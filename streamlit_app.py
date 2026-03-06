"""
COVID-19 Mortality Prediction — Streamlit Application
=====================================================
Four-tab app: Executive Summary, Descriptive Analytics,
Model Performance, Explainability & Interactive Prediction.

Usage:  streamlit run streamlit_app.py
"""

import os, json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

# ---- Config ----
st.set_page_config(page_title="COVID-19 Mortality Prediction", layout="wide",
                   page_icon="🦠")
BASE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
MODELS = os.path.join(BASE, "models")
METRICS_DIR = os.path.join(BASE, "metrics")
DATA = os.path.join(BASE, "data")

# ---- Load Artifacts ----
@st.cache_data
def load_metrics():
    with open(os.path.join(METRICS_DIR, "metrics.json")) as f:
        return json.load(f)

@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODELS, name))



@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(DATA, "scaler.pkl"))

@st.cache_resource
def load_explainer():
    return joblib.load(os.path.join(MODELS, "shap_explainer.pkl"))

data = load_metrics()
metrics = data["metrics"]
stats = data["stats"]
feature_names = data["feature_names"]
scaler = load_scaler()

# ---- Styling ----
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1a1a2e;
                   margin-bottom: 0.5rem;}
    .sub-header {font-size: 1.1rem; color: #555; margin-bottom: 1.5rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  padding: 1.2rem; border-radius: 12px; color: white;
                  text-align: center; margin-bottom: 1rem;}
    .metric-card h3 {margin: 0; font-size: 2rem;}
    .metric-card p {margin: 0; font-size: 0.85rem; opacity: 0.85;}
    .insight-box {background: #f0f4ff; border-left: 4px solid #667eea;
                  padding: 1rem; border-radius: 6px; margin: 0.8rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🦠 COVID-19 Mortality Prediction Dashboard</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">End-to-end data science pipeline: from exploratory analysis to predictive modeling, explainability, and interactive prediction.</p>',
            unsafe_allow_html=True)

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Executive Summary",
    "📊 Descriptive Analytics",
    "🤖 Model Performance",
    "🔍 Explainability & Prediction"
])

# ===== TAB 1: Executive Summary ==========================================
with tab1:
    st.header("Executive Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
            <h3>{stats['n_rows']:,}</h3><p>Total Patients</p></div>""",
            unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <h3>{stats['n_features']}</h3><p>Features</p></div>""",
            unsafe_allow_html=True)
    with col3:
        death_pct = stats['death_counts'].get('1', stats['death_counts'].get(1, 0)) / stats['n_rows'] * 100
        st.markdown(f"""<div class="metric-card">
            <h3>{death_pct:.1f}%</h3><p>Mortality Rate</p></div>""",
            unsafe_allow_html=True)
    with col4:
        best_model = max(metrics, key=lambda k: metrics[k].get("F1", 0))
        best_f1 = metrics[best_model]["F1"]
        st.markdown(f"""<div class="metric-card">
            <h3>{best_f1:.3f}</h3><p>Best F1 Score</p></div>""",
            unsafe_allow_html=True)

    st.subheader("📌 About the Dataset")
    st.write("""
    This dataset contains **over 1 million patient records** from Mexico's COVID-19 surveillance
    system. Each row represents a patient tested for COVID-19, with information about their
    demographics, pre-existing medical conditions (comorbidities), and clinical outcomes.
    """)

    st.subheader("🎯 Prediction Task")
    st.write("""
    **Target variable:** `DEATH` — binary indicator (1 = patient died, 0 = survived).

    **Why this matters:** Predicting mortality risk enables hospitals to triage patients more
    effectively, allocate ICU beds and ventilators to those most at risk, and design targeted
    interventions for vulnerable populations. Early identification of high-risk patients can
    directly save lives.
    """)

    st.subheader("📊 Feature Overview")
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Type": ["Binary" if f != "AGE" else "Numerical" for f in feature_names],
        "Description": [
            "Patient sex (0=Female, 1=Male)",
            "Whether patient was hospitalized",
            "Pneumonia diagnosis",
            "Patient age in years",
            "Whether patient is pregnant",
            "Diabetes diagnosis",
            "Chronic Obstructive Pulmonary Disease",
            "Asthma diagnosis",
            "Immunosuppressed condition",
            "Hypertension diagnosis",
            "Other chronic disease",
            "Cardiovascular disease",
            "Obesity diagnosis",
            "Chronic renal disease",
            "Tobacco use",
            "COVID-19 positive test result",
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.subheader("🔑 Key Findings")
    st.write(f"""
    - **Best performing model:** {best_model} with F1 = {best_f1:.4f}
    - **Critical risk factors:** Age, pneumonia, hospitalization status, and diabetes
      are the strongest predictors of mortality.
    - **Class imbalance:** The dataset is imbalanced (mortality rate ~{death_pct:.1f}%),
      which we handle through stratified splitting and evaluation with F1 score rather
      than raw accuracy.
    - **SHAP analysis** reveals that pneumonia and advanced age have the strongest
      positive impact on predicted mortality risk.
    """)

# ===== TAB 2: Descriptive Analytics ======================================
with tab2:
    st.header("Descriptive Analytics")
    st.write("Visual exploration of the COVID-19 patient dataset before modeling.")

    # Target Distribution
    st.subheader("1. Target Variable Distribution")
    st.image(os.path.join(PLOTS, "target_distribution.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> The dataset is <strong>imbalanced</strong> — the majority
    of patients survived. This class imbalance is expected in real-world medical data
    and necessitates using stratified splits and F1/AUC metrics rather than plain accuracy.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Age Distribution
    st.subheader("2. Age Distribution by Mortality")
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(PLOTS, "age_violin.png"))
    with col2:
        st.image(os.path.join(PLOTS, "age_histogram.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> Deceased patients are significantly older on average.
    The violin plot shows that the mortality group has a median age around 60–65,
    while survivors cluster around 35–45. Age is clearly one of the strongest
    risk factors.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Comorbidity Mortality Rates
    st.subheader("3. Mortality Rate by Comorbidity")
    st.image(os.path.join(PLOTS, "comorbidity_mortality.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> Patients with COPD, renal chronic disease, cardiovascular
    disease, and diabetes have the highest mortality rates among comorbidity groups.
    Immunosuppression and hypertension also show elevated risk. This aligns with
    medical literature on COVID-19 vulnerability factors.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Hospitalization & Pneumonia
    st.subheader("4. Hospitalization & Pneumonia vs Mortality")
    st.image(os.path.join(PLOTS, "hospitalized_pneumonia_mortality.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> Hospitalized patients and those diagnosed with
    pneumonia show dramatically higher mortality rates. Pneumonia in particular
    is a strong indicator — patients with pneumonia have a mortality rate several
    times higher than those without, reflecting the severity of respiratory
    complications in COVID-19 deaths.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # COVID Positive vs Mortality
    st.subheader("5. COVID Test Result vs Mortality")
    st.image(os.path.join(PLOTS, "covid_positive_mortality.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> COVID-positive patients show a different mortality
    profile compared to those with negative or pending results, highlighting
    the direct impact of confirmed COVID-19 infection on patient outcomes.
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Correlation Heatmap
    st.subheader("6. Correlation Heatmap")
    st.image(os.path.join(PLOTS, "correlation_heatmap.png"))
    st.markdown("""<div class="insight-box">
    <strong>Insight:</strong> The strongest correlations with DEATH include AGE,
    PNEUMONIA, and HOSPITALIZED. Notable feature–feature correlations include
    AGE–HYPERTENSION and AGE–DIABETES (older patients more likely to have these
    comorbidities). Most comorbidities show modest positive correlations with each
    other, suggesting patients often have multiple conditions simultaneously.
    </div>""", unsafe_allow_html=True)

# ===== TAB 3: Model Performance ==========================================
with tab3:
    st.header("Model Performance Comparison")

    # Model comparison table
    st.subheader("📊 Metrics Summary")
    display_cols = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
    rows = []
    for name, m in metrics.items():
        row = {col: m.get(col, "N/A") for col in display_cols}
        row["Model"] = name
        bp = m.get("best_params", None)
        row["Best Hyperparameters"] = str(bp) if bp else "N/A"
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    comp_df = comp_df[["Model"] + display_cols + ["Best Hyperparameters"]]
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # F1 comparison bar chart
    st.subheader("📈 F1 Score Comparison")
    st.image(os.path.join(PLOTS, "model_comparison_f1.png"))

    st.divider()

    # ROC curves
    st.subheader("📉 ROC Curves (All Models)")
    st.image(os.path.join(PLOTS, "roc_curves_all.png"))

    st.divider()

    # Decision Tree Visualization
    st.subheader("🌳 Decision Tree Visualization")
    st.image(os.path.join(PLOTS, "decision_tree_viz.png"))
    st.caption("Visualization of the best Decision Tree (truncated to depth 3 for readability).")

    st.divider()

    # MLP Training History
    st.subheader("🧠 Neural Network Training History")
    st.image(os.path.join(PLOTS, "mlp_training_history.png"))
    st.caption("Training loss curve over iterations (sklearn MLPClassifier).")

    st.divider()

    # Commentary
    st.subheader("💡 Analysis")
    best_model_name = max(metrics, key=lambda k: metrics[k].get("F1", 0))
    st.write(f"""
    **Best model: {best_model_name}** with an F1 score of
    {metrics[best_model_name]['F1']:.4f} and AUC-ROC of
    {metrics[best_model_name]['AUC-ROC']:.4f}.

    **Key observations:**
    - **Ensemble and boosted models** (Random Forest, Gradient Boosting) generally outperform
      the simpler Logistic Regression baseline and individual Decision Tree, as expected.
    - **Neural Network (MLP)** achieves competitive performance but requires more
      computational resources and offers less interpretability.
    - **Logistic Regression** provides a reasonable baseline, which is valuable for its
      simplicity and interpretability.
    - **Trade-offs:** While Gradient Boosting/Random Forest achieve the best raw metrics, Logistic
      Regression is far more interpretable. The Decision Tree offers a middle ground with
      visual explainability. For a clinical setting, the choice between models would depend
      on whether explainability or marginal accuracy gains are more valued.
    """)

# ===== TAB 4: Explainability & Interactive Prediction =====================
with tab4:
    st.header("Explainability & Interactive Prediction")

    shap_col, pred_col = st.columns([1, 1])

    with shap_col:
        st.subheader("🔬 SHAP Feature Importance")

        shap_tab1, shap_tab2, shap_tab3 = st.tabs([
            "Beeswarm", "Bar Plot", "Waterfall"])

        with shap_tab1:
            st.image(os.path.join(PLOTS, "shap_summary_beeswarm.png"))
            st.write("""
            The beeswarm plot shows each feature's impact on model predictions.
            **Red = high feature value, Blue = low.** Features at the top have
            the largest impact. High age and presence of pneumonia push predictions
            strongly toward mortality.
            """)
        with shap_tab2:
            st.image(os.path.join(PLOTS, "shap_bar.png"))
            st.write("""
            Mean absolute SHAP values rank features by overall importance.
            This tells us which features the model relies on most, regardless
            of direction.
            """)
        with shap_tab3:
            st.image(os.path.join(PLOTS, "shap_waterfall.png"))
            st.write("""
            Waterfall plot for a single high-risk patient, showing how each
            feature contributes to pushing the prediction above or below the
            base value (average prediction).
            """)

        st.subheader("🧠 Interpretation")
        st.write("""
        **Key findings from SHAP analysis:**

        1. **AGE** is the most important predictor — older patients have dramatically
           higher predicted mortality risk.
        2. **PNEUMONIA** is the second most impactful — its presence strongly increases
           mortality predictions.
        3. **HOSPITALIZED** status also has significant predictive power.
        4. **Comorbidities** like diabetes, hypertension, and obesity contribute
           moderately to risk.

        **Implications for decision-makers:** These insights can guide triage protocols.
        Hospitals should prioritize monitoring for older patients with pneumonia and
        multiple comorbidities. Public health campaigns could target diabetes and
        hypertension management as protective measures.
        """)

    with pred_col:
        st.subheader("🎯 Interactive Prediction")
        st.write("Adjust the patient features below and get a real-time mortality risk prediction.")

        # Model selector
        model_choice = st.selectbox(
            "Select Model",
            ["Logistic Regression", "Decision Tree", "Random Forest",
             "Gradient Boosting", "Neural Network (MLP)"]
        )

        st.divider()

        # Input features
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.slider("Age", 0, 120, 50)
            sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            pneumonia = st.selectbox("Pneumonia", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hospitalized = st.selectbox("Hospitalized", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            covid_positive = st.selectbox("COVID Positive", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            obesity = st.selectbox("Obesity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        with col_b:
            pregnant = st.selectbox("Pregnant", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            copd = st.selectbox("COPD", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            asthma = st.selectbox("Asthma", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            immunosuppression = st.selectbox("Immunosuppression", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            other_disease = st.selectbox("Other Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            cardiovascular = st.selectbox("Cardiovascular", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            renal_chronic = st.selectbox("Renal Chronic", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            tobacco = st.selectbox("Tobacco Use", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

        # Build input DataFrame in correct column order
        input_data = pd.DataFrame({
            "SEX": [sex], "HOSPITALIZED": [hospitalized], "PNEUMONIA": [pneumonia],
            "AGE": [age], "PREGNANT": [pregnant], "DIABETES": [diabetes],
            "COPD": [copd], "ASTHMA": [asthma], "IMMUNOSUPPRESSION": [immunosuppression],
            "HYPERTENSION": [hypertension], "OTHER_DISEASE": [other_disease],
            "CARDIOVASCULAR": [cardiovascular], "OBESITY": [obesity],
            "RENAL_CHRONIC": [renal_chronic], "TOBACCO": [tobacco],
            "COVID_POSITIVE": [covid_positive]
        })
        input_data = input_data[feature_names]  # ensure column order

        # Scaled version for LR and MLP
        input_scaled = input_data.copy()
        input_scaled["AGE"] = scaler.transform(input_data[["AGE"]])

        if st.button("🔮 Predict", use_container_width=True, type="primary"):
            # Load model and predict
            if model_choice == "Logistic Regression":
                mdl = load_model("logistic_regression.pkl")
                prob = mdl.predict_proba(input_scaled)[:, 1][0]
                pred = int(prob >= 0.5)
            elif model_choice == "Decision Tree":
                mdl = load_model("decision_tree.pkl")
                prob = mdl.predict_proba(input_data)[:, 1][0]
                pred = int(prob >= 0.5)
            elif model_choice == "Random Forest":
                mdl = load_model("random_forest.pkl")
                prob = mdl.predict_proba(input_data)[:, 1][0]
                pred = int(prob >= 0.5)
            elif model_choice == "Gradient Boosting":
                mdl = load_model("gradient_boosting.pkl")
                prob = mdl.predict_proba(input_data)[:, 1][0]
                pred = int(prob >= 0.5)
            else:  # MLP
                mdl = load_model("mlp_model.pkl")
                prob = mdl.predict_proba(input_scaled)[:, 1][0]
                pred = int(prob >= 0.5)

            # Display result
            st.divider()
            if pred == 1:
                st.error(f"⚠️ **High Risk — Predicted: DEATH**")
                st.metric("Mortality Probability", f"{prob*100:.1f}%")
            else:
                st.success(f"✅ **Low Risk — Predicted: SURVIVED**")
                st.metric("Mortality Probability", f"{prob*100:.1f}%")

            # SHAP waterfall for this specific input
            st.divider()
            st.subheader("🔍 SHAP Explanation for This Prediction")
            try:
                explainer = load_explainer()
                sv = explainer.shap_values(input_data)
                # GradientBoostingClassifier returns single array; others may return list
                if isinstance(sv, list):
                    shap_row = sv[1][0]
                    base_val = explainer.expected_value[1]
                else:
                    shap_row = sv[0]
                    base_val = float(explainer.expected_value)
                explanation = shap.Explanation(
                    values=shap_row,
                    base_values=base_val,
                    data=input_data.values[0],
                    feature_names=feature_names,
                )
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.plots.waterfall(explanation, show=False)
                plt.title("SHAP Waterfall — Your Custom Input")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning(f"SHAP waterfall not available for this model type: {e}")
                st.info("SHAP waterfall is generated using the Gradient Boosting model explainer.")
