# 5G User Prediction — AI Course Group Project

> **Course**: Artificial Intelligence | **Department**: Software Engineering  
> **Deliverables**: Jupyter Notebook · Analysis Report (Word) · Presentation PPT  
> **Metric**: AUC (Area Under the ROC Curve)

---

## Project Overview

This project predicts whether a telecom user is a **5G user** (binary classification) based on user profile and communication-related data — billing, data usage, activity patterns, plan type, and regional information.

We implement and compare **four machine learning models**:

| Model | Type | Expected AUC |
|-------|------|-------------|
| Logistic Regression | Linear baseline | ~0.74 |
| Random Forest | Bagging ensemble | ~0.87 |
| XGBoost | Gradient boosting | ~0.905 |
| LightGBM | Efficient boosting | ~0.912 |

The dataset contains **60 features** (20 categorical + 38 numerical) covering billing amount, monthly data usage, call minutes, SMS count, and plan details.

---

## Project Structure

```
5G-user-prediction/
├── 5G用户预测.ipynb              # Main notebook: EDA, 4 models, visualizations
├── .gitignore
└── README.md                     # This file
```

---

## Quick Start

### 1. Prerequisites

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm
```

Or install Jupyter Notebook for a better experience:

```bash
pip install notebook
```

### 2. Get the Data

Download `train.csv` from the course cloud platform and place it in the project root directory. The notebook **auto-detects** the data file — if missing, it generates synthetic demonstration data on-the-fly so you can still run every cell and produce all charts.

### 3. Run the Notebook

```bash
jupyter notebook 5G用户预测.ipynb
```

Then click **Kernel → Restart & Run All**.

Alternatively, open the `.ipynb` file directly in **VS Code** with the Jupyter extension installed.

After execution, the following charts are generated automatically:

- `target_distribution.png` — Target variable distribution (pie + bar)
- `feature_distribution.png` — Feature distributions by 5G/non-5G user class
- `model_comparison.png` — ROC curves + AUC bar chart comparison
- `confusion_matrix.png` — Confusion matrices for LR vs RF

---

## Notebook Sections

| Section | Content |
|---------|---------|
| 1 | Import dependencies |
| 2 | Data loading (auto-detect real CSV or generate synthetic) |
| 3 | Exploratory Data Analysis (EDA) — distributions, missing values |
| 4 | Feature preprocessing — Label Encoding, scaling, train/val split |
| 5 | Model training — LR, RF, XGBoost\*, LightGBM\* |
| 6 | Model comparison — ROC curves, AUC comparison table |
| 7 | Evaluation function (as required by the assignment) |
| 8 | Confusion matrix analysis |
| 9 | Summary & improvement directions |

> \* XGBoost & LightGBM run only if installed; skipped gracefully otherwise.

---

## Team Division

A 5-member team, each responsible for one model + corresponding report section + one PPT slide.

| Role | Model | Responsibilities |
|------|-------|-----------------|
| Leader | Coordination | Code integration, final execution, PPT presentation |
| Member 2 | Logistic Regression | LR code + tuning, EDA distribution charts |
| Member 3 | Random Forest | RF code + feature importance, data preprocessing |
| Member 4 | XGBoost | XGB code + hyperparameter tuning, ROC curves |
| Member 5 | LightGBM | LGB code + early stopping, AUC comparison table |

All members collaborate on report formatting, PPT polishing, and final AUC verification.

---

## Approaches & Rationale

**Why four models?**

- **Logistic Regression** — Simple, interpretable baseline. Required by the assignment as a traditional AI approach.
- **Random Forest** — Robust to feature noise and class imbalance. Provides feature importance ranking for analysis.
- **XGBoost** — Industry-standard gradient boosting with regularization, preventing overfitting on high-dimensional data.
- **LightGBM** — Leaf-wise growth strategy and histogram-based splitting for faster training on large datasets.

**Why AUC as the metric?**

AUC is threshold-invariant and works well with imbalanced binary classification — exactly the scenario for 5G user prediction where positive samples may be the minority.

---

## Key Findings

- Billing amount (`num_0`) and monthly data usage (`num_1`) are the **most predictive** numerical features
- Tree-based ensemble models (RF, XGBoost, LightGBM) significantly outperform the linear baseline
- LightGBM achieves the highest AUC with the fastest training time

---

## Improvement Directions

1. **Hyperparameter tuning** — GridSearchCV or Optuna for systematic optimization
2. **Feature engineering** — Cross-features (e.g., billing-to-plan-rent ratio as upgrade-intent signal)
3. **Class imbalance** — SMOTE oversampling or class weights for extreme imbalance scenarios
4. **Model fusion** — Stacking/Blending ensemble of the top-performing models
5. **Alternative metric** — AUC-PR for precision-recall trade-off when positive class is rare
