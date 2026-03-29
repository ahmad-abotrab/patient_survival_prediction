# In-Hospital Mortality Prediction from Structured Clinical Data

> **Can we predict in-hospital mortality from structured patient records, and which features contribute most to risk estimation?**

This project is a hands-on machine learning study using real ICU data to predict whether a patient will survive their hospital stay. It was built as a learning journey through the full ML pipeline — from raw data exploration all the way to model evaluation and interpretability.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Column Groups](#column-groups)
4. [Stage 1 — Exploratory Data Analysis (EDA)](#stage-1--exploratory-data-analysis-eda)
5. [Stage 2 — Preprocessing & Feature Engineering](#stage-2--preprocessing--feature-engineering)
6. [Stage 3 — Baseline & Advanced Models](#stage-3--baseline--advanced-models)
7. [Stage 4 — Evaluation](#stage-4--evaluation)
8. [Stage 5 — Interpretability](#stage-5--interpretability)
9. [Stage 6 — Human-in-the-Loop Simulation](#stage-6--human-in-the-loop-simulation)
10. [Key Learnings](#key-learnings)
11. [Project Structure](#project-structure)
12. [How to Run](#how-to-run)

---

## Project Overview

This project applies supervised binary classification to predict **in-hospital mortality** (`hospital_death`) for ICU patients. The data includes demographics, vital signs, lab results, APACHE severity scores, and comorbidity flags.

The goal is not just to train a model, but to understand:
- What the data looks like and what challenges it presents
- Which models are appropriate for a medical binary classification problem
- How to evaluate models properly when classes are imbalanced
- How to explain model predictions to build trust

---

## Dataset

| Property | Value |
|---|---|
| Source | WiDS Datathon (Kaggle) |
| File | `data/patients_data.csv` |
| Rows | 91,713 patients |
| Columns | 84 features + 1 target |
| Target | `hospital_death` (binary: 0 = survived, 1 = died) |

### Class Distribution (Target)

| Class | Count | Percentage |
|---|---|---|
| 0 — Survived | 83,798 | **91.37%** |
| 1 — Died | 7,915 | **8.63%** |

> **Important:** The dataset is heavily imbalanced (~10:1 ratio). This means accuracy is a misleading metric — a model that always predicts "survived" would score 91.37% accuracy while being completely useless. We must use AUC-ROC, F1, and Recall as the primary metrics.

---

## Column Groups

| Group | Columns | Description |
|---|---|---|
| **IDs** | `encounter_id`, `patient_id`, `hospital_id`, `icu_id` | Identifiers — excluded from modeling |
| **Target** | `hospital_death` | What we are predicting |
| **Demographics** | `age`, `bmi`, `gender`, `ethnicity`, `height`, `weight` | Patient background |
| **Admission Context** | `icu_admit_source`, `icu_stay_type`, `icu_type`, `pre_icu_los_days`, `elective_surgery` | How and why the patient was admitted |
| **APACHE Scores** | `apache_2_diagnosis`, `apache_3j_diagnosis`, `apache_post_operative`, `arf_apache`, `gcs_*`, `heart_rate_apache`, `map_apache`, `resprate_apache`, `temp_apache`, `intubated_apache`, `ventilated_apache` | ICU severity scoring system variables |
| **Day 1 Vitals** | `d1_diasbp_*`, `d1_heartrate_*`, `d1_mbp_*`, `d1_resprate_*`, `d1_spo2_*`, `d1_sysbp_*`, `d1_temp_*` | Min/max vital signs during first 24 hours |
| **Hour 1 Vitals** | `h1_diasbp_*`, `h1_heartrate_*`, `h1_mbp_*`, `h1_resprate_*`, `h1_spo2_*`, `h1_sysbp_*` | Min/max vital signs during first hour |
| **Lab Results** | `d1_glucose_max/min`, `d1_potassium_max/min` | Day 1 lab values |
| **APACHE Probabilities** | `apache_4a_hospital_death_prob`, `apache_4a_icu_death_prob` | Pre-computed clinical risk scores |
| **Comorbidities** | `aids`, `cirrhosis`, `diabetes_mellitus`, `hepatic_failure`, `immunosuppression`, `leukemia`, `lymphoma`, `solid_tumor_with_metastasis` | Binary flags for chronic conditions |
| **Body Systems** | `apache_3j_bodysystem`, `apache_2_bodysystem` | Categorical body system classification |

---

## Stage 1 — Exploratory Data Analysis (EDA)

**Notebook:** `notebook/loading_data.ipynb`

### What we did

- Loaded the dataset and inspected its shape, types, and first rows
- Analyzed the target variable distribution
- Built a complete missing-values summary
- Explored basic statistics for numerical and categorical columns

### Key Findings

#### Missing Values

| Column | Missing % | Notes |
|---|---|---|
| `d1_potassium_max/min` | ~10.45% | Lab values not always collected |
| `h1_mbp_noninvasive_max/min` | ~9.90% | Not all patients have noninvasive measurements |
| `apache_4a_hospital_death_prob` | ~8.67% | Pre-computed clinical score, sometimes unavailable |
| `apache_4a_icu_death_prob` | ~8.67% | Same source as above |
| `temp_apache` | ~4.47% | Some patients missing temperature reading |
| `age`, `bmi`, `height`, `weight` | ~1–4% | Standard demographic missingness |
| `encounter_id`, `hospital_id`, `hospital_death` | 0% | Complete columns |

> Most missing values are under 10% — manageable with median/mode imputation. The `apache_4a_*` columns are special: they are pre-computed clinical risk scores and must be treated carefully to avoid data leakage.

#### Demographics Summary

| Feature | Range | Mean | Notes |
|---|---|---|---|
| `age` | 16 – 89 | 62.3 years | Older population as expected in ICU |
| `bmi` | 14.8 – 67.8 | 29.2 | Average overweight range |
| `height` | 137 – 195 cm | 169.6 cm | Normal distribution |
| `weight` | 38.6 – 186 kg | 84 kg | Wide range |
| `gender` | M / F | — | 53.9% Male |
| `ethnicity` | 6 categories | Caucasian (78%) | Dominant group, class imbalance in ethnicity |

#### Comorbidity Prevalence

| Condition | Prevalence |
|---|---|
| Diabetes Mellitus | 22.5% |
| Solid Tumor w/ Metastasis | 2.1% |
| Immunosuppression | 2.6% |
| Cirrhosis | 1.6% |
| Hepatic Failure | 1.3% |
| Leukemia | 0.7% |
| Lymphoma | 0.4% |
| AIDS | 0.09% |

#### ICU Admission Types

- Most common admission source: **Accident & Emergency** (54,060 patients)
- Most common ICU stay type: **admit** (86,183 patients)
- Most common ICU type: **Med-Surg ICU** (50,586 patients)

---

## Stage 2 — Preprocessing & Feature Engineering

### Steps

1. **Drop ID columns** — `encounter_id`, `patient_id`, `hospital_id`, `icu_id` carry no predictive signal

2. **Handle missing values**
   - Numerical columns → median imputation
   - Categorical columns → mode imputation (or "Unknown" category)

3. **Encode categorical variables**
   - `gender`, `ethnicity`, `icu_admit_source`, `icu_stay_type`, `icu_type`, `apache_3j_bodysystem`, `apache_2_bodysystem` → Label Encoding or One-Hot Encoding

4. **Handle class imbalance** using one or more of:
   - `class_weight='balanced'` in model parameters
   - SMOTE (Synthetic Minority Over-sampling Technique)
   - Adjusting classification threshold

5. **Feature scaling**
   - Tree-based models (Decision Tree, Random Forest) → no scaling needed
   - Logistic Regression and KNN → StandardScaler

6. **Train/Test split** — 80% train, 20% test, stratified by target

7. **Consider leakage risk** — `apache_4a_hospital_death_prob` and `apache_4a_icu_death_prob` are pre-computed clinical scores that already encode death probability, so their impact should be monitored carefully during experimentation.

---

## Stage 3 — Baseline & Advanced Models

We experiment with multiple models, starting simple and progressively increasing complexity.

### Models Tested

| # | Model                         | Why We Try It                                                           |
|---|-------------------------------|-------------------------------------------------------------------------|
| 1 | **Logistic Regression**       | Simple, interpretable baseline. Works well as a starting point.         |
| 2 | **Decision Tree**             | Easy to visualize and understand decisions. Can overfit.                |
| 3 | **Random Forest**             | Ensemble of trees — handles missing values better, reduces overfitting. |
| 4 | **K-Nearest Neighbors (KNN)** | Instance-based learning, useful as a distance-based comparison model.   |

### Hyperparameter Tuning

For the models, we tune using `RandomizedSearchCV` with cross-validation:
- Decision Tree: `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`
- KNN: `n_neighbors`, `weights`, `metric`

---

## Stage 4 — Evaluation

### Why Accuracy Alone Is Not Enough

With 91.37% of patients surviving, a dummy model that always predicts "0" achieves 91.37% accuracy. We need metrics that reflect performance on the **minority class (deaths)**.

### Metrics Used

| Metric | What It Measures |
|---|---|
| **Accuracy** | Overall fraction of correct predictions (kept for reference only). |
| **ROC-AUC** | Overall ability to rank positives above negatives. Main metric. |
| **Macro F1** | F1 averaged across both classes, treating each class equally. |
| **Recall (Sensitivity)** | Of all actual deaths, how many did we catch? (Critical in medical context.) |
| **Precision** | Of all predicted deaths, how many were real deaths? |
| **Confusion Matrix** | Full breakdown of TP, FP, TN, FN. |

### Model Comparison (Notebook Results)

| Model | Accuracy | Macro F1 | ROC-AUC | Precision (Died) | Recall (Died) |
|---|---|---|---|---|---|
| Logistic Regression | 0.8028 | 0.6412 | 0.8761 | 0.2715 | 0.7631 |
| Decision Tree | 0.8003 | 0.6346 | 0.8565 | 0.2641 | 0.7353 |
| Random Forest | 0.9091 | 0.7192 | 0.8826 | 0.4747 | 0.5028 |
| KNN | 0.9178 | 0.6203 | 0.6802 | 0.5728 | 0.1889 |

---

## Stage 5 — Interpretability

Understanding *why* a model makes a prediction is essential in healthcare. A model that can't explain itself will not be trusted by clinicians.

### Tools & Techniques

1. **Feature Importance (Tree-based)**
   - Built-in `.feature_importances_` from Random Forest
   - Shows which features the model relies on most globally

2. **SHAP (SHapley Additive exPlanations)**
   - Explains individual predictions: "why did the model predict death for this specific patient?"
   - SHAP summary plots show global importance
   - SHAP waterfall plots explain a single patient's prediction
   - Expected features with high SHAP values: `apache_4a_hospital_death_prob`, GCS scores, age, MAP, SpO2

3. **Partial Dependence Plots (PDP)**
   - Shows how changing one feature (e.g., age) affects the predicted risk
   - Helps understand the relationship direction and non-linearity

---

## Stage 6 — Human-in-the-Loop Simulation

This stage simulates how a clinician would interact with the model in practice.

### Concept

- Given a new patient's data, the model outputs a **death probability score**
- If the score is above a threshold (e.g., 0.5), the patient is flagged as high-risk
- A SHAP explanation is generated showing the top 5 factors driving the prediction
- The clinician reviews the explanation and can override the decision

### What We Explore

- How does changing the threshold affect Precision vs. Recall?
- At what threshold do we maximize recall (catch most deaths) while keeping false alarms manageable?
- What features do clinicians find most actionable?

---

## Key Learnings

Through building this project, the following concepts were practiced and reinforced:

- **Class imbalance** is one of the most common real-world challenges — accuracy is not a reliable metric
- **Feature leakage** is subtle — the APACHE probability scores are computed from the same data, so using them can make models look better than they really are
- **Random Forest** provided the best balance between discrimination and minority-class performance in this notebook run
- **Recall matters more than Precision** in medical mortality prediction — missing a death (False Negative) is more dangerous than a false alarm
- **SHAP values** bridge the gap between black-box models and clinical trust
- **Preprocessing matters** — median imputation for vitals, handling categorical variables correctly, and stratified splitting all affect results significantly

---

## Project Structure

```
patient_survival_prediction/
│
├── data/
│   ├── patients_data.csv          # Main cleaned dataset
│   └── dataset.csv.zip            # Original compressed dataset
│
├── notebook/
│   └── loading_data.ipynb         # EDA + preprocessing + model training + evaluation
│
└── readme.md                      # Project report (this file)
```

---

## How to Run

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap imbalanced-learn
```

### Run the Notebook

```bash
cd notebook/
jupyter notebook loading_data.ipynb
```

---

## Dataset Reference

> WiDS (Women in Data Science) Datathon Dataset — Patient Survival Prediction
> Available on Kaggle: `https://www.kaggle.com/competitions/widsdatathon2020`

---

*Built as a learning project exploring the full machine learning pipeline on a real-world clinical dataset.*
