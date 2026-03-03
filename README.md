# COVID-19 Mortality Prediction

End-to-end data science project: Exploratory Data Analysis → Predictive Modeling → Model Explainability → Streamlit Deployment.

## Dataset

`covid_cleaned3.csv` — Over 1 million COVID-19 patient records from Mexico's surveillance system. Binary classification task predicting patient mortality (DEATH).

**Features:** SEX, HOSPITALIZED, PNEUMONIA, AGE, PREGNANT, DIABETES, COPD, ASTHMA, IMMUNOSUPPRESSION, HYPERTENSION, OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO, COVID_POSITIVE

## Setup

```bash
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
python3 train_models.py
```

This will:
- Generate all EDA visualizations in `plots/`
- Train 5 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, MLP)
- Perform SHAP analysis
- Save all models to `models/` and metrics to `metrics/`

## Launch Streamlit App

```bash
streamlit run streamlit_app.py
```

The app has 4 tabs:
1. **Executive Summary** — Dataset overview and key findings
2. **Descriptive Analytics** — EDA visualizations with interpretations
3. **Model Performance** — Comparison table, ROC curves, training history
4. **Explainability & Prediction** — SHAP analysis + interactive prediction tool

## Project Structure

```
├── covid_cleaned3.csv        # Dataset
├── train_models.py           # Training pipeline
├── streamlit_app.py          # Streamlit web app
├── requirements.txt          # Dependencies
├── README.md                 # This file
├── models/                   # Saved trained models
├── plots/                    # Generated visualizations
├── metrics/                  # Model metrics (JSON/CSV)
└── data/                     # Preprocessing artifacts (scaler)
```

## Models

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline classifier |
| Decision Tree | 5-fold GridSearchCV over max_depth, min_samples_leaf |
| Random Forest | 5-fold GridSearchCV over n_estimators, max_depth |
| XGBoost | 5-fold GridSearchCV over n_estimators, max_depth, learning_rate |
| Neural Network (MLP) | Keras MLP with 3 hidden layers, dropout, Adam optimizer |

All models use `random_state=42`.
