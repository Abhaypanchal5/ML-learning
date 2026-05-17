# ML Learning Journey — XGBoost (Gradient Boosting)

Four binary classification problems using XGBoost — the algorithm that wins more Kaggle competitions than any other single model. Unlike Random Forest where trees vote independently, XGBoost builds each tree specifically to fix the errors the previous trees made.

---

## Repo Structure

```
README.md
notebooks/
  employee_promotion.ipynb   ← HR — Promotion Prediction
  insurance_fraud.ipynb      ← Finance — Fraud Detection
  loan_default.ipynb         ← Banking — Default Prediction
  telecom_churn.ipynb        ← Telecom — Churn Prediction
data/
  employee_promotion.csv
  insurance_fraud.csv
  loan_default.csv
  telecom_churn.csv
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
```

---

## The Core Concept — Gradient Boosting

Random Forest builds 100 independent trees and takes a majority vote. XGBoost builds trees sequentially — each one learning specifically from where the previous ones went wrong.

```
Tree 1  →  makes predictions, calculates errors
Tree 2  →  focuses on the errors Tree 1 made
Tree 3  →  focuses on remaining errors
...
Final   →  sum of all trees combined
```

This is gradient descent applied to tree building. Instead of adjusting weights, XGBoost adds new trees that point in the direction of the gradient (steepest error reduction).

### The Objective Function

$$\text{Obj} = \underbrace{\sum_{i=1}^{n} L(y_i, \hat{y}_i)}_{\text{Training Loss}} + \underbrace{\sum_{k=1}^{K} \Omega(f_k)}_{\text{Regularisation}}$$

The regularisation term is built into the maths — not a post-hoc parameter. This is why XGBoost is harder to overfit than plain gradient boosting.

---

## Early Stopping — The Right Way to Set n_estimators

Instead of guessing the number of trees, early stopping monitors validation loss and stops automatically when it stops improving:

```python
model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
```

`early_stopping_rounds=20` — if validation loss doesn't improve for 20 consecutive trees, training stops. The model uses only the trees up to the best iteration.

### Preserving Column Names After Scaling

When passing a numpy array from `scaler.fit_transform()` to XGBoost, column names are lost and features appear as `f0`, `f1`, `f2` in importance outputs. Fix with:

```python
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)
```

---

## Three Types of Feature Importance

```python
for imp_type in ['weight', 'gain', 'cover']:
    imp = model.get_booster().get_score(importance_type=imp_type)
    # weight  = how many times a feature was used in splits
    # gain    = average improvement when that feature is used
    # cover   = average number of rows affected by that feature
```

**Gain is the most meaningful for analysts** — it measures actual predictive contribution, not just frequency of use.

---

## The Full Pipeline Used Across All Notebooks

```
Load Data
    ↓
Explore (shape, info, describe, nulls, duplicates)
    ↓
Copy
    ↓
Fix Data Types (regex + astype)
    ↓
Impossible Values → bulk loop for negatives + column-specific upper bounds
    ↓
Outliers (IQR on continuous columns, exclude binary/categorical/target)
    ↓
Drop Duplicates
    ↓
Fill Nulls (floor for integers, round for continuous)
    ↓
Define X and y
    ↓
Train/Test Split 80/20
    ↓
StandardScaler (preserve column names as DataFrame)
    ↓
Train XGBClassifier with early stopping
    ↓
Print best_iteration
    ↓
Predict + classification_report
    ↓
All three importance types (weight, gain, cover)
    ↓
Gap check
    ↓
3-plot summary (Confusion Matrix, Learning Curve, Gain Importance)
```

---

---

## Project 1 — Employee Promotion Prediction

**File:** `employee_promotion.ipynb`
**Dataset:** 30,300 rows → 30,000 after cleaning
**Question:** Can we predict which employees will be promoted in the next cycle?

### Dataset

| Column | Description |
|---|---|
| `age` | Employee age |
| `years_at_company` | Tenure in years |
| `performance_score` | Performance rating (0–100) |
| `training_hours` | Annual training hours |
| `awards_won` | Recognition awards received |
| `avg_review_score` | Average performance review (0–100) |
| `num_reports` | Number of direct reports |
| `dept_kpi_score` | Department KPI achievement (1–10) |
| `prev_promoted` | Binary — previously promoted |
| `promoted` | **Target** — 1 = promoted, 0 = not |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `performance_score` | String with `_err` suffix |
| Impossible | All columns except age | Negative values — bulk loop |
| Impossible | `age` | Below 18 and above 100 |
| Impossible | `performance_score` | 150 — caught via IQR |
| Impossible | `avg_review_score` | 150 — caught via IQR |
| Impossible | `dept_kpi_score` | 20 (max is 10) — caught via IQR |
| Nulls | All columns except target | ~600 per column |
| Duplicates | All columns | 298 exact copies |

### Early Stopping Result

Best iteration = 79 out of 200. The model converged at 79 trees — adding more would have increased overfitting without improving test accuracy.

### Results

```
Accuracy:   0.7557
Gap:        0.0318
Best iter:  79
```

### Feature Importance Note

Feature names appeared as `f0`, `f1`, `f2` in importance output because scaling returned a numpy array instead of a DataFrame. The gain ranking showed f4 and f8 as dominant — these correspond to `awards_won` and `prev_promoted` based on column position. Previous recognition and prior promotion history are the strongest signals for future promotion — a finding consistent with real HR research on promotion patterns.

---

---

## Project 2 — Insurance Fraud Detection

**File:** `insurance_fraud.ipynb`
**Dataset:** 33,300 rows → 33,000 after cleaning
**Question:** Can we flag fraudulent insurance claims before they're paid out?

### Dataset

| Column | Description |
|---|---|
| `claim_amount` | Total claim value |
| `policy_age_years` | How long the policy has been active |
| `prev_claims` | Number of previous claims |
| `injury_severity` | Injury severity rating (1–4) |
| `days_to_report` | Days between incident and claim filing |
| `num_witnesses` | Number of witnesses present |
| `vehicle_age_years` | Age of vehicle |
| `repair_cost` | Estimated repair cost |
| `agent_flagged` | Binary — flagged by agent |
| `is_fraud` | **Target** — 1 = fraud, 0 = legitimate |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `claim_amount` | String with `_err` suffix |
| Impossible | All columns | Negative values — bulk loop |
| Impossible | `injury_severity` | 10 (max is 4) — caught via IQR |
| Impossible | `num_witnesses` | 99 — caught via IQR |
| Nulls | All columns except target | ~661 per column |
| Duplicates | All columns | 298 exact copies |

### Learning Rate Issue

`learning_rate=0.4` was used — significantly higher than the standard 0.01–0.2 range. Early stopping triggered at only 11 trees. The model took large gradient steps and converged prematurely without exploring the loss surface properly.

```
Best iteration: 11 (out of 500)
```

This explains the lowest accuracy in the batch. At `learning_rate=0.1`, early stopping would likely find an optimum around 80–120 trees with substantially better accuracy.

**Rule going forward:** keep `learning_rate` between 0.05 and 0.2. Let early stopping handle the tree count.

### Results

```
Accuracy:   0.7327  ← affected by high learning rate
Gap:        0.0228
Best iter:  11      ← premature convergence
```

### Key Domain Note

For fraud detection, Recall for class 1 matters most — missing a fraudulent claim costs the company the full claim amount. The current model's Recall of 0.73 means 27% of actual fraud goes undetected. Lowering the probability threshold from 0.5 to 0.35 and fixing the learning rate would both improve this meaningfully.

---

---

## Project 3 — Loan Default Prediction

**File:** `loan_default.ipynb`
**Dataset:** 32,300 rows → 32,000 after cleaning
**Question:** Can we predict loan defaults before approving applications?

### Dataset

| Column | Description |
|---|---|
| `loan_amount` | Total loan value |
| `annual_income` | Borrower's annual income |
| `credit_score` | Credit bureau score (300–850) |
| `loan_term_years` | Loan repayment period |
| `employment_years` | Years in current employment |
| `num_open_loans` | Active loan count |
| `debt_to_income` | Debt-to-income ratio |
| `collateral_value` | Value of collateral offered |
| `missed_payments` | Historical missed payment count |
| `defaulted` | **Target** — 1 = defaulted, 0 = repaid |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `credit_score` | String with `_err` suffix |
| Impossible | All columns | Negative values — bulk loop |
| Impossible | `credit_score` | 1,200 — caught with explicit `> 900` rule |
| Impossible | `debt_to_income` | 5.0 — caught via IQR |
| Nulls | All columns except target | ~641 per column |
| Duplicates | All columns | 300 exact copies |

### Best Impossible Values Section in Batch

```python
for col in Columns:
    df_copy.loc[df_copy[col]<0, col] = np.nan
df_copy.loc[df_copy['credit_score']>900, 'credit_score'] = np.nan
```

Only notebook in the batch with a domain-specific explicit upper bound rule alongside the bulk negative check. Credit score of 1200 is physically impossible — the explicit cap at 900 catches it cleanly.

### Best Imputation Split in Batch

```python
Round = ['loan_amount', 'annual_income', 'employment_years',
         'debt_to_income', 'collateral_value']
Floor = ['credit_score', 'loan_term_years', 'num_open_loans', 'missed_payments']
```

Every column correctly categorised. Continuous financial values in Round, integer counts and scores in Floor.

### Results — Best in Batch

```
Accuracy:   0.8300  ← best in batch
Gap:        0.0321
Best iter:  111
```

### Feature Importance — Top Finding

`missed_payments` dominated gain importance at 63.47 — nearly double the second-ranked feature. This is the most decisive single-feature finding across all four notebooks. A borrower who has missed payments before is the clearest signal of future default risk — the model independently discovered what every credit risk officer already knows.

`credit_score` ranked second in gain — the traditional primary metric in lending — but less than half as important as payment history. This is consistent with real-world credit research showing that behavioural signals (what you've done) outpredict profile signals (who you are on paper).

---

---

## Project 4 — Telecom Customer Churn

**File:** `telecom_churn.ipynb`
**Dataset:** 31,300 rows → 31,000 after cleaning
**Question:** Can we predict which customers will cancel before they do?

### Dataset

| Column | Description |
|---|---|
| `tenure_months` | Months as a customer |
| `monthly_charges` | Monthly bill |
| `total_charges` | Cumulative charges paid |
| `num_services` | Active service count |
| `support_calls` | Support calls made |
| `contract_type` | 0 = monthly, 1 = 1yr, 2 = 2yr |
| `payment_delays` | Late payment count |
| `data_usage_gb` | Monthly data usage |
| `num_complaints` | Formal complaints raised |
| `churned` | **Target** — 1 = left, 0 = stayed |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `monthly_charges` | String with `_err` suffix |
| Impossible | All columns | Negative values — bulk loop |
| Impossible | `monthly_charges` | 9,999 — caught via IQR |
| Nulls | All columns except target | ~622 per column |
| Duplicates | All columns | 299 exact copies |

### Outlier Exclusion — Most Thoughtful in Batch

```python
Columns.remove('contract_type')   # categorical
Columns.remove('churned')         # target
Columns.remove('num_complaints')  # count capped at 5
```

Three correct removals — including `num_complaints` which was excluded because its range (0–5) has no meaningful outliers. Shows domain-aware exclusion logic rather than mechanical application.

### Imputation Split — Cleanest in Batch

```python
Round = ['monthly_charges', 'total_charges', 'data_usage_gb']
Floor = ['tenure_months', 'num_services', 'support_calls',
         'contract_type', 'payment_delays', 'num_complaints']
```

Every column correctly placed. `total_charges` correctly in Round despite being derived from integer × float — the product is continuous. `tenure_months` correctly in Floor — you don't have 36.7 months of tenure.

`total_charges` accumulated 855 nulls — more than other columns — because it's derived from `tenure_months × monthly_charges`. Outlier removal on the derived column was broader, correctly handled with median imputation.

### Results

```
Accuracy:   0.8210
Gap:        0.0276
Best iter:  122
```

### Feature Importance Insight

`support_calls` topped gain importance at 40.89. Customers who call support repeatedly are signalling dissatisfaction before they formally churn. This is a well-known pattern in telecom — the support queue is an early warning system for churn. The model found this signal independently from data, which validates both the model and the business intuition.

`payment_delays` ranked second at 38.22 — customers who delay payments are economically disengaged before they formally cancel. Together, support calls and payment delays explain the two most actionable pre-churn signals a retention team can act on.

---

---

## Results Across All Four

| Project | Accuracy | Gap | Best Iter | Top Gain Feature |
|---|---|---|---|---|
| Employee Promotion | 0.7557 | 0.0318 | 79 | awards_won (f4) |
| Insurance Fraud | 0.7327 | 0.0228 | **11** ⚠️ | prev_claims (f2) |
| Loan Default | **0.8300** | 0.0321 | 111 | missed_payments |
| Telecom Churn | 0.8210 | 0.0276 | 122 | support_calls |

Insurance Fraud had the lowest accuracy due to `learning_rate=0.4` — early stopping triggered at only 11 trees. Loan Default had the best accuracy with the most domain-correct feature importance finding.

---

## What XGBoost Does Differently

**Sequential not parallel** — unlike Random Forest where all trees are independent, XGBoost trees build on each other. Each tree is a direct correction of previous errors.

**Early stopping replaces n_estimators tuning** — no manual loop from 10 to 200 trees needed. Set n_estimators high, set early_stopping_rounds=20, and let the algorithm find the optimum automatically. The best_iteration tells you exactly how many trees were actually needed.

**Three importance types** — Random Forest gives one `feature_importances_` score. XGBoost gives weight (frequency), gain (quality), and cover (breadth). Gain is the analyst-relevant metric — it measures actual predictive contribution.

**Built-in regularisation** — L1 and L2 penalties are part of the objective function. XGBoost is harder to overfit than plain Gradient Boosting because overfitting is penalised at the maths level, not just through parameter choices.

---

## What's Next

- Cross-validation — more reliable model evaluation than a single train/test split
- GridSearchCV — systematic hyperparameter tuning
- LightGBM — faster XGBoost alternative for very large datasets
- Feature Engineering — creating better input columns

---

## Stack

Python 3 · Pandas · NumPy · XGBoost · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
