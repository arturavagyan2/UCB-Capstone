# Santander Product Recommendation — Capstone (Final)

## Overview
This capstone builds a **Top‑N product recommendation system** for Santander customers using the Kaggle **Santander Product Recommendation** dataset.  
The business goal is to recommend products a customer is likely to **add next month** (cross‑sell targeting).

The submission includes:
- **Recommenders (SURPRISE):** SVD (primary) + KNNBasic (comparison)
- **Baselines:** popularity by additions / popularity by ownership
- **Evaluation:** Precision@K, Recall@K, HitRate@K, Coverage using a strict hold‑out month
- **Interpretability:** a separate supervised adoption model (LogReg) with **Permutation Feature Importance** for `credit_card`

## Data
- File: `train_ver2.csv` (Kaggle)
- Expected path: `data/train_ver2.csv`
- Snapshot approach: **last two months only**
  - Train month = second‑to‑last snapshot
  - Test month = last snapshot
- Ground truth: **next‑month additions** (0→1 between train and test snapshot)

### Dataset profile (this run)
Customers in both months: **695,232**  
Average # products owned (train): **1.332** (median = 1; 75% ≤ 2; max = 14)  
Average # products added next month: **0.0385**  
% customers adding ≥1 product next month: **2.998%**  
Total additions in hold‑out month: **26,773**

### Product prevalence (train month; top examples)
- `current_account`: **60.6%**
- `direct_debit`: **12.0%**
- `particular_account`: **10.9%**
- `e_account`: **8.0%**
- `payroll_account`: **7.8%**
- `credit_card`: **3.76%**

> Interpretation: Portfolios are small and skewed toward a few common products (especially `current_account`). Next‑month additions are sparse, making this a difficult ranking problem.

## Notebook
- `Santander_Recommender.ipynb`
  - Memory‑safe loading and optional sampling via `MAX_CUSTOMERS`
  - Column renaming to readable English names
  - EDA + visualizations (prevalence, portfolio size distribution, additions distribution)
  - Baselines and SURPRISE recommenders
  - Negative sampling for implicit feedback (ownership = 1, sampled unowned = 0)
  - Ranking‑aligned tuning using **HitRate@K**
  - Evaluation for:
    - **All users** (population-level performance)
    - **Users with ≥1 addition** (diagnostic ranking performance)

## Models included
### Baselines
1. **Popularity by additions**: recommend the products most frequently added (excluding already owned).
2. **Popularity by ownership**: recommend the products most commonly owned (excluding already owned).

### Recommendation models (SURPRISE)
1. **SVD** (matrix factorization) — primary recommender
2. **KNNBasic** (item‑based collaborative filtering) — comparison recommender

### Interpretability model (separate from recommender)
- **Logistic Regression** with preprocessing (imputation + one‑hot encoding) and CV tuning
- **Permutation Feature Importance** (F1-based) for `credit_card` next‑month adoption

## Results summary (this run)
### Baselines (Top‑7)
**All users**
- Popularity (by additions): Precision@7 = **0.0054**, Recall@7 = **0.0292**, HitRate@7 = **0.0292**, Coverage = **87.5% (21/24)**
- Popularity (by ownership): Precision@7 = **0.0053**, Recall@7 = **0.0285**, HitRate@7 = **0.0286**, Coverage = **87.5% (21/24)**

**Users with ≥1 addition (N = 20,843)**
- Popularity (by additions): Precision@7 = **0.1791**, Recall@7 = **0.9727**, HitRate@7 = **0.9755**, Coverage = **87.5% (21/24)**
- Popularity (by ownership): Precision@7 = **0.1757**, Recall@7 = **0.9509**, HitRate@7 = **0.9551**, Coverage = **87.5% (21/24)**

### SURPRISE recommenders (Top‑7; negative sampling)
Final training interactions (implicit + negatives): **16,685,568** rows.

**All users**
- **SVD:** Precision@7 = **0.0053**, Recall@7 = **0.0289**, HitRate@7 = **0.0290**, Coverage = **87.5% (21/24)**
- **KNNBasic:** Precision@7 = **0.0011**, Recall@7 = **0.0063**, HitRate@7 = **0.0074**, Coverage = **62.5% (15/24)**

**Users with ≥1 addition (N = 20,843)**
- **SVD:** Precision@7 = **0.1776**, Recall@7 = **0.9628**, HitRate@7 = **0.9660**, Coverage = **87.5% (21/24)**
- **KNNBasic:** Precision@7 = **0.0356**, Recall@7 = **0.2086**, HitRate@7 = **0.2454**, Coverage = **62.5% (15/24)**

> Interpretation: In this run, **SVD performs on par with the strongest popularity baseline** (and far above KNNBasic) at both population and conditional levels. This suggests the catalog/additions are heavily concentrated in a small set of popular products, and personalization beyond popularity is limited under a two‑month setup. Extending the time horizon and adding richer signals is the most promising path to improvement.

### Interpretability (credit card adoption)
- Positive rate (credit card additions): **~0.46%** (3,202 positives out of 695,232)
- Best CV F1: **0.0417** (LogReg, `C=1.0`, `class_weight='balanced'`)
- PR‑AUC (Average Precision, in-sample): **0.0424**
- Confusion matrix (in-sample): `[[553343, 138687], [141, 3061]]`
  - Class 1 recall ≈ **0.96**, precision ≈ **0.02**

**Permutation feature importance (Top drivers for `credit_card`)**
1. `account_start_date` (largest importance)
2. `relationship_type_desc_month`
3. `age`
4. `is_new_customer`
5. `province_code`

> Interpretation: Onboarding timing/tenure proxies and relationship attributes are the strongest predictors of credit card adoption. Because of extreme class imbalance, this model is configured to favor recall; a deployment would tune thresholds to improve precision for campaign targeting.

## How to run
1. This project uses the Kaggle dataset from the Santander competition:
  - Kaggle data page: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data
  Download `train_ver2.csv` from Kaggle and place it in:
  - `data/train_ver2.csv`

2. Use Python 3.11 (recommended) and install:
   - `numpy<2.0`, `scikit-surprise`, `pandas`, `scikit-learn`, `matplotlib`
3. Run the notebook: `Capstone_Final_Santander_Recommender.ipynb`

## Next steps
- Use more months and temporal features to improve personalization beyond last-month ownership.
- Add eligibility rules and contact policy constraints for deployable recommendations.
- Validate impact with an online A/B test (conversion uplift, incremental revenue).
- Explore implicit-ranking objectives (pairwise ranking/BPR) and additional models (SVDpp/NMF).
