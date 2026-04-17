# ML Learning Journey — Logistic Regression Practice

Four classification problems across four completely different domains. Same algorithm, different data, different real-world stakes. The goal was to get comfortable with the full logistic regression pipeline and understand that evaluation means more than just reading the accuracy number.

---

## Repo Structure

```
README.md                          ← you are here
notebooks/
  disease_diagnosis.ipynb          ← Healthcare — Disease Prediction
  employee_attrition.ipynb         ← HR — Employee Churn
  ipl_match_win.ipynb              ← Cricket — Match Outcome
  loan_default.ipynb               ← Banking — Loan Default
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

---

## The Core Concept — Why Logistic Regression

Linear Regression predicts a number. Logistic Regression predicts a category.

The key is the sigmoid function, which squashes any output between 0 and 1:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where z is the same linear combination from before:

$$z = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n$$

The output is a probability. Anything above 0.5 gets predicted as class 1, anything below as class 0. That 0.5 threshold can be adjusted based on domain needs.

**Positive coefficient** → feature increases probability of Yes  
**Negative coefficient** → feature decreases probability of Yes

---

## Evaluation Metrics — The Full Picture

Four metrics matter here. Accuracy alone is not enough.

**Accuracy**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** — of everything predicted Yes, how many were actually Yes?
$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** — of all actual Yes cases, how many did the model catch?
$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** — harmonic mean of precision and recall
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

Which metric matters most depends entirely on the domain. That's one of the main things these four notebooks taught me.

---

## The Standard Pipeline Used Across All Notebooks

```
Load Data
    ↓
Explore (shape, dtypes, nulls, duplicates, describe)
    ↓
Copy Original — never touch the raw file
    ↓
Fix Data Types
    ↓
Handle Impossible Values → nullify
    ↓
Handle Outliers (IQR) → nullify
    ↓
Drop Duplicates
    ↓
Fill Nulls (median imputation)
    ↓
Define X and y
    ↓
Train Test Split 80/20
    ↓
Train LogisticRegression
    ↓
Predict class + probability
    ↓
Evaluate — classification_report + confusion matrix
    ↓
Visualise — probability distribution, confusion matrix, coefficients
```

---

---

## Project 1 — Disease Diagnosis

**File:** `disease_diagnosis.ipynb`  
**Domain:** Healthcare  
**Dataset:** 634 rows, 8 columns  
**Question:** Can we predict whether a patient has a disease based on basic clinical measurements?

### Features

| Column | Description |
|---|---|
| `age` | Patient age |
| `bmi` | Body mass index |
| `blood_sugar` | Blood glucose level |
| `blood_pressure` | Systolic blood pressure |
| `cholesterol` | Cholesterol reading |
| `is_smoker` | Binary — 1 if smoker |
| `family_history` | Binary — 1 if family history of disease |
| `has_disease` | **Target** — 1 = has disease, 0 = healthy |

### Dirty Data Found

| Problem | Column | What It Was |
|---|---|---|
| Wrong data type | `bmi` | Stored as string with `_err` suffix |
| Impossible value | `age` | Value of 150 |
| Impossible value | `blood_pressure` | Negative value |
| Outlier | `blood_sugar` | Value of 900 |
| Outlier | `age` | Filtered values above 80 |
| Nulls | `blood_sugar`, `cholesterol` | 20–21 missing each |
| Duplicates | All columns | 14 exact duplicates |

### Key Technical Choice — Pipeline with StandardScaler

This was the only notebook where a sklearn `Pipeline` was used, combining feature scaling and the model into one object:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])
pipeline.fit(X_train, y_train)
```

Logistic Regression is sensitive to features on different scales. Blood sugar ranges from 70–300 while BMI ranges from 16–45. Scaling puts them on equal footing before the model sees them. Good practice for any logistic regression model.

### Split

```python
train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
```

`stratify=y` ensures both train and test sets have the same proportion of class 0 and class 1. Important when class distribution matters.

### Results

```
Accuracy:   0.911
Precision:  0.91 (class 1)
Recall:     0.99 (class 1)
F1 Score:   0.95 (class 1)
```

### Confusion Matrix

```
              Predicted No    Predicted Yes
Actual No         15               10
Actual Yes         1               98
```

### What the Model Learned

The model caught 98 out of 99 actual disease cases — Recall of 0.99. For a healthcare screening model that's the right priority. Missing a sick patient is far worse than an unnecessary follow-up checkup. The 10 false positives (healthy people flagged as sick) are acceptable in this context.

### Key Observation

Recall matters most here. A false negative in disease detection means a sick person goes home untreated. This model prioritises that correctly.

---

---

## Project 2 — Employee Attrition

**File:** `employee_attrition.ipynb`  
**Domain:** Human Resources  
**Dataset:** 612 rows, 8 columns  
**Question:** Can we predict which employees are likely to leave the company?

### Features

| Column | Description |
|---|---|
| `satisfaction_score` | Employee satisfaction rating (1–5) |
| `salary_lakh` | Annual salary in lakhs |
| `years_at_company` | Tenure in years |
| `promotions_last_5yr` | Number of promotions received |
| `weekly_hours` | Average hours worked per week |
| `num_projects` | Number of active projects |
| `manager_rating` | Manager effectiveness rating (1–5) |
| `left_company` | **Target** — 1 = left, 0 = stayed |

### Dirty Data Found

| Problem | Column | What It Was |
|---|---|---|
| Wrong data type | `salary_lakh` | Stored as object (string with `_err`) |
| Impossible value | `satisfaction_score` | Value of 9 (max is 5) |
| Impossible value | `weekly_hours` | Value of 200 |
| Impossible value | `years_at_company` | Negative value (-3) |
| Impossible value | `salary_lakh` | Negative value |
| Nulls | `satisfaction_score`, `manager_rating` | 20 missing each |
| Duplicates | All columns | 12 exact duplicates |

### Coefficient Interpretation

```
satisfaction_score     → -1.7519  (strongest predictor — unhappy employees leave)
manager_rating         → -0.8413  (bad manager = higher exit risk)
promotions_last_5yr    → -0.7670  (no growth = more likely to leave)
years_at_company       → -0.2552  (longer tenure = more likely to stay)
num_projects           → +0.3680  (overloaded employees leave more)
weekly_hours           → +0.1509  (overworked employees leave more)
salary_lakh            → -0.1150  (higher salary = slight retention effect)
```

All coefficients match real HR intuition. Satisfaction score being the strongest negative predictor makes sense — people leave managers and cultures, not companies.

### Results

```
Accuracy:   0.9417
Precision:  0.94
Recall:     0.9216
F1 Score:   0.9307
```

Cleanest and most balanced result across all four notebooks. Both precision and recall above 0.92 means the model is performing well in both directions — not over-predicting leavers and not missing them either.

### Key Observation

For HR attrition, both precision and recall matter. False positives waste retention budget on employees who weren't leaving. False negatives let valuable employees walk out without intervention. F1 Score is the right metric here — and 0.93 is strong.

---

---

## Project 3 — IPL Match Win Prediction

**File:** `ipl_match_win.ipynb`  
**Domain:** Sports Analytics (Cricket)  
**Dataset:** 593 rows, 8 columns  
**Question:** Can we predict match outcome at the end of the powerplay (first 6 overs)?

### Features

| Column | Description |
|---|---|
| `runs_first_6_overs` | Runs scored in powerplay |
| `wickets_lost_pp` | Wickets lost in powerplay |
| `current_run_rate` | Run rate at end of powerplay |
| `boundary_count` | Number of 4s and 6s |
| `home_venue` | Binary — 1 if playing at home |
| `top3_batsmen_in` | Binary — 1 if top 3 still batting |
| `extras_conceded` | Wides, no-balls etc |
| `match_won` | **Target** — 1 = won, 0 = lost |

### Dirty Data Found

| Problem | Column | What It Was |
|---|---|---|
| Wrong data type | `current_run_rate` | Stored as string with `_err` |
| Impossible value | `wickets_lost_pp` | Negative value (-2) |
| Outlier | `runs_first_6_overs` | 500 runs in 6 overs (impossible) |
| Impossible value | `current_run_rate` | Negative value |
| Outlier | `extras_conceded` | Value of 300 |
| Nulls | `current_run_rate`, `boundary_count` | 18 and 17 missing |
| Duplicates | All columns | 13 duplicates (after outlier nullification) |

### Bug Found

In the feature selection cell, `runs_first_6_overs` was listed twice and `wickets_lost_pp` was accidentally excluded:

```python
# Bug — same column listed twice, wickets missing
X = df_copy[['runs_first_6_overs',
             'runs_first_6_overs',   ← duplicate
             'current_run_rate', ...]]
```

This means the model never saw wicket information — one of the most important powerplay features in cricket. This explains the lower performance compared to other notebooks.

### Results

```
Accuracy:   0.698
Precision:  0.7416
Recall:     0.8462
F1 Score:   0.7904
```

Weakest result of the four notebooks. The bug above is the primary reason — removing a key feature from X directly limits what the model can learn. Fixing the feature list would likely push accuracy above 0.80.

### Key Observation

The model still picked up useful signal despite the bug. Recall of 0.85 means it catches most winning situations correctly. But 23 false positives out of 38 actual losses suggests it's too optimistic — predicting wins when the team actually lost. More features (especially wickets) would fix this.

---

---

## Project 4 — Loan Default Prediction

**File:** `loan_default.ipynb`  
**Domain:** Banking / Finance  
**Dataset:** 665 rows, 7 columns  
**Question:** Can we predict whether a loan applicant will default?

### Features

| Column | Description |
|---|---|
| `age` | Applicant age |
| `annual_income` | Annual income in rupees |
| `loan_amount` | Total loan requested |
| `credit_score` | Credit bureau score (300–850) |
| `employment_years` | Years in current employment |
| `existing_loans` | Number of active loans |
| `defaulted` | **Target** — 1 = defaulted, 0 = repaid |

### Dirty Data Found

| Problem | Column | What It Was |
|---|---|---|
| Wrong data type | `age` | Stored as object (string with `_err`) |
| Impossible value | `age` | Negative value (-10) |
| Impossible value | `credit_score` | Value of 1200 (max is 850) |
| Impossible value | `annual_income` | Negative value (-5000) |
| Impossible value | `loan_amount` | Zero value |
| Nulls | `credit_score`, `employment_years` | 20 missing each |
| Duplicates | All columns | 15 exact duplicates |

### Coefficient Interpretation

```
existing_loans        → +1.5110  (most existing debt = highest default risk)
employment_years      → -0.4310  (stable employment = lower risk)
age                   → -0.1058  (older applicants slightly less risky)
credit_score          → -0.0211  (higher score = lower risk — correct direction)
annual_income         → ~0.0000  (coefficient too small — scaling issue)
loan_amount           → ~0.0000  (same issue)
```

The near-zero coefficients for `annual_income` and `loan_amount` are a scaling problem. These columns have values in the lakhs (100,000+) while others are in single digits. Without StandardScaler, the model struggles to compare them fairly. Adding a pipeline with StandardScaler (like Project 1 did) would fix this.

### Results

```
Accuracy:   0.9462
Precision:  0.8947
Recall:     0.7727
F1 Score:   0.8293
```

### Class Imbalance Note

Only 13% of applicants in this dataset defaulted. This means even a model that predicts "no default" every time would score 87% accuracy. The meaningful number here is Recall — 0.77 means the model catches 77% of actual defaults. For a bank, the remaining 23% of missed defaults is still a significant risk. Future improvement: class weighting or oversampling.

### Key Observation

In banking, a missed default (false negative) costs far more than a wrongly rejected good applicant (false positive). Recall should be the primary metric here, not accuracy. Adjusting the decision threshold from 0.5 to 0.3 would catch more defaults at the cost of more false positives — a worthwhile trade in a lending context.

---

---

## Results Summary

| Project | Domain | Accuracy | Precision | Recall | F1 | Key Metric |
|---|---|---|---|---|---|---|
| Disease Diagnosis | Healthcare | 0.911 | 0.91 | 0.99 | 0.95 | Recall |
| Employee Attrition | HR | 0.942 | 0.94 | 0.92 | 0.93 | F1 Score |
| IPL Match Win | Cricket | 0.698 | 0.74 | 0.85 | 0.79 | Accuracy |
| Loan Default | Banking | 0.946 | 0.89 | 0.77 | 0.83 | Recall |

---

## Things Learned Across All Four

**Metric selection is domain-specific.** The same F1 Score of 0.83 means very different things in healthcare vs cricket. Context determines which errors are acceptable.

**Feature selection bugs are silent.** The cricket notebook accidentally excluded wickets from X and the model still ran without error — just with worse results. Always verify `X.columns` before training.

**StandardScaler matters for logistic regression.** The near-zero coefficients in the loan notebook on income and loan amount happened because those columns were on a completely different scale. The disease notebook avoided this by using a Pipeline.

**Class imbalance affects how you read accuracy.** The loan dataset was 87% non-defaults. High accuracy there doesn't mean the model is doing useful work — checking Recall tells the real story.

**Probability scores are more useful than hard predictions.** Every notebook included `predict_proba()` alongside `predict()`. The probability lets you adjust the threshold based on how risk-tolerant the use case is — something a binary 0/1 output can't do.

---

## What's Next

- Decision Tree — non-linear classification, feature importance, visualising the actual decision logic
- Random Forest — ensemble method, generally stronger than a single tree
- K-Means Clustering — unsupervised grouping without labels

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
