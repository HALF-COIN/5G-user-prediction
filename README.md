# 5G User Prediction — AI Course Group Project

> **Course**: Artificial Intelligence | **Department**: Software Engineering  
> **Deliverables**: Jupyter Notebook · Analysis Report (Word) · Presentation PPT  
> **Metric**: AUC (Area Under the ROC Curve)

---

## Project Overview

This project predicts whether a telecom user is a **5G user** (binary classification) based on user profile and communication-related data — billing, data usage, activity patterns, plan type, and regional information.

We implement and compare **four machine learning models**:

| Model | Type | AUC (Real Data) |
|-------|------|-----------------|
| Logistic Regression | Linear baseline | Failed (predicted 0 positives) |
| Random Forest | Bagging ensemble | Breakthrough — 1,775 real 5G users found |
| **XGBoost** 🏆 | Gradient boosting | **0.9154** (Best) |
| LightGBM | Efficient boosting | 0.872 |

The dataset contains **839,993 rows × 60 features** (20 categorical + 38 numerical + 1 ID + 1 target). Key finding: only **1.3%** of users are 5G users — extreme class imbalance (1 : 74.5).

---

## Project Structure

```
5G-user-prediction/
├── 5G用户预测.ipynb                         # Main notebook: EDA, 4 models, visualizations
├── .gitignore
└── README.md                                # This file
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

- **Extreme class imbalance**: Only 1.3% of users are 5G users (1:74.5 ratio). This makes AUC-PR a valuable complementary metric alongside AUC-ROC.
- **Top predictive features**: `num_37` (r=+0.12), `num_3` (r=+0.096), `num_4` — these are the strongest individual predictors of 5G adoption.
- **Logistic Regression failed** on this highly imbalanced dataset — it predicted zero positive cases (recall=0, precision=0). Linear models require careful class-weight tuning for extreme imbalance.
- **XGBoost achieved the highest AUC (0.9154)** with `scale_pos_weight` handling class imbalance effectively. Strong recall on the minority class.
- **Random Forest broke through** — despite only ~11,000 positive samples, RF correctly identified 1,775 real 5G users thanks to `class_weight='balanced'`.
- **LightGBM (AUC=0.872)** underperformed expectations — its leaf-wise growth may be less robust than XGBoost's regularization under extreme class imbalance.

---

## Improvement Directions

1. **Class imbalance strategies** — SMOTE oversampling, ADASYN, or cost-sensitive learning could help LR and LightGBM handle the 1:74.5 imbalance better.
2. **Feature engineering** — Domain-specific cross-features (e.g., billing-to-plan-rent ratio, data usage per contract type) could surface hidden 5G upgrade signals stronger than raw features.
3. **Hyperparameter tuning** — Optuna or Bayesian optimization for scale_pos_weight, learning rate, and tree depth tuning on XGBoost/LightGBM.
4. **Stacking ensemble** — Use LR as meta-learner over RF + XGBoost + LGB base models to compound their strengths.
5. **Alternative metrics** — AUC-PR (Precision-Recall AUC) is more sensitive than AUC-ROC when positive class is only 1.3%.
