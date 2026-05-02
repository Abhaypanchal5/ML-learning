# ML Learning Journey — Support Vector Machine (SVM)

Four binary classification problems using SVM — the model built around finding the maximum margin boundary between classes. Unlike tree-based models that vote, SVM finds the single most confident decision boundary possible given the training data.

---

## Repo Structure

```
README.md
notebooks/
  cancer_diagnosis.ipynb    ← Healthcare — Tumour Classification
  email_spam.ipynb          ← Tech — Spam Detection
  student_pass_fail.ipynb   ← Education — Pass/Fail Prediction
  wine_quality.ipynb        ← Food Science — Quality Classification
data/
  cancer_diagnosis.csv
  email_spam.csv
  student_pass_fail.csv
  wine_quality.csv
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
```

---

## The Core Concept — Maximum Margin

Every classification model draws a boundary between classes. Logistic Regression finds any line that separates them. SVM finds the one with the maximum gap between the boundary and the nearest points on each side. That gap is called the **margin**.

```
          Class 0          |  Margin  |          Class 1
               ●           |          |           ■
             ●             |          |             ■
               ●           | ←  W  → |           ■
```

The points sitting on the edge of the margin are called **support vectors** — the only training points that actually determine where the boundary goes. All other points are irrelevant to the final boundary.

### The Maths

$$\text{Minimise} \quad \frac{1}{2}||w||^2 \quad \text{subject to} \quad y_i(w \cdot x_i + b) \geq 1$$

Maximising margin = minimising `||w||`. A constrained quadratic optimisation problem.

### The Kernel Trick

When data isn't linearly separable in 2D, SVM transforms it into a higher dimension where it becomes separable — without explicitly computing the transformation. The RBF kernel was used across all four notebooks:

```python
kernel='rbf'   # radial basis function — handles curved boundaries
```

### The C Parameter

Controls the tradeoff between margin width and misclassification tolerance:

```
Low C  →  wide margin, allows some errors  →  better generalisation
High C →  narrow margin, fewer errors allowed  →  overfitting risk
```

---

## Why Pipeline Is Mandatory Here

SVM uses distance calculations. Features on different scales will dominate the boundary unfairly. Scaling must happen inside the Pipeline — not before splitting — to prevent the scaler from seeing test data during fit:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale',
                   random_state=42, probability=True))
])
```

---

## The C Tuning Loop

Run in every notebook before training the final model:

```python
C_values = [0.01, 0.1, 1, 10, 100]
for c in C_values:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('svm', SVC(C=c, kernel='rbf', random_state=42, gamma='scale'))])
    pipe.fit(X_train, y_train)
    # record train and test scores

best_index = np.argmax(test_scores)
best_c = C_values[best_index]
plt.axvline(best_c, linestyle='--', label=f'Best C={best_c}')
```

The vertical line marks the optimal C. The final model was always trained at this value.

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
Impossible Values → nullify
    ↓
Outliers (IQR on selected columns) → nullify
    ↓
Drop Duplicates
    ↓
Fill Nulls (floor for counts, round for continuous)
    ↓
Define X and y
    ↓
Train/Test Split 80/20
    ↓
C Tuning Loop (0.01, 0.1, 1, 10, 100) with axvline
    ↓
Train Pipeline(StandardScaler + SVC) at optimal C
    ↓
Predict class + probability
    ↓
classification_report + confusion matrix
    ↓
Gap check (train vs test accuracy)
    ↓
3-plot summary (confusion matrix, C tuning, train vs test bars)
```

---

---

## Project 1 — Cancer Diagnosis

**File:** `cancer_diagnosis.ipynb`
**Dataset:** 6,050 rows → 6,000 after cleaning
**Question:** Can we classify a tumour as malignant or benign from cell nucleus measurements?

### Dataset

| Column | Description |
|---|---|
| `radius_mean` | Mean radius of cell nuclei |
| `texture_mean` | Standard deviation of grey-scale values |
| `perimeter_mean` | Mean perimeter of nuclei |
| `area_mean` | Mean area of nuclei |
| `smoothness_mean` | Local variation in radius lengths |
| `compactness_mean` | Perimeter² / area − 1.0 |
| `concavity_mean` | Severity of concave portions |
| `symmetry_mean` | Symmetry of nuclei |
| `malignant` | **Target** — 1 = malignant, 0 = benign |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `radius_mean` | String with `_err` suffix |
| Impossible | `radius_mean`, `texture_mean`, `symmetry_mean` | Negative values |
| Impossible | `area_mean` | 99,999 (caught via IQR) |
| Impossible | `smoothness_mean` | 2.0 — proportion must be ≤ 1 (caught via IQR) |
| Nulls | All columns except target | ~120 per column |
| Duplicates | All columns | 50 exact copies |

### Model

```python
Pipeline([StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)])
```

### Results

```
Accuracy:   0.8450
Precision:  0.85 (benign), 0.83 (malignant)
Recall:     0.84 (benign), 0.85 (malignant)
Gap:        0.0127  ← excellent generalisation
```

### Domain Note

Recall for malignant tumours (class 1) = 0.85. This means 15% of actual malignant cases were classified as benign. In a clinical screening context, this false negative rate is too high — a missed malignant tumour delays treatment. Lowering the probability threshold from 0.5 to 0.3 would increase Recall at the cost of more false positives (unnecessary biopsies). In cancer diagnostics, that trade-off is almost always worth making.

SVM was historically one of the first algorithms applied to cancer classification in the 1990s — the maximum margin approach works well because malignant and benign cells tend to be well-separated in feature space.

---

---

## Project 2 — Email Spam Detection

**File:** `email_spam.ipynb`
**Dataset:** 7,050 rows → 7,000 after cleaning
**Question:** Can we classify emails as spam or legitimate based on content patterns?

### Dataset

| Column | Description |
|---|---|
| `word_freq_free` | Frequency of the word "free" |
| `word_freq_money` | Frequency of the word "money" |
| `word_freq_click` | Frequency of the word "click" |
| `char_freq_exclaim` | Frequency of `!` characters (0–1) |
| `char_freq_dollar` | Frequency of `$` characters (0–1) |
| `capital_run_avg` | Average length of capital letter runs |
| `capital_run_long` | Longest capital letter run |
| `num_links` | Number of hyperlinks in email |
| `sender_reputation` | Sender reputation score (0–10) |
| `is_spam` | **Target** — 1 = spam, 0 = legitimate |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `capital_run_avg` | String with `_err` suffix |
| Impossible | `word_freq_free`, `word_freq_money`, `num_links` | Negative values |
| Impossible | `sender_reputation` | 15 (max is 10) — caught via IQR |
| Outlier | `capital_run_avg` | 500 — caught via IQR |
| Nulls | All columns except target | ~140 per column |
| Duplicates | All columns | 50 exact copies |

### Upgrade — Best C Tuning Implementation

```python
best_index = np.argmax(test_scores)
best_c = C_values[best_index]
plt.axvline(best_c, linestyle='--', label=f'Best C={best_c}')
```

Best C found programmatically and plotted with a vertical line. Found C=1 as optimal and correctly used in the final model.

### Results

```
Accuracy:   0.8186
Precision:  0.84 (spam), 0.80 (not spam)
Recall:     0.79 (spam), 0.85 (not spam)
Gap:        0.0370  ← within acceptable range
```

### Domain Note

For a spam filter, Precision matters more than Recall. A false positive (legitimate email marked as spam) is more disruptive than a false negative (spam reaching the inbox). Precision of 0.84 means 16% of emails flagged as spam are actually legitimate — acceptable for most deployments but worth monitoring. Raising the decision threshold from 0.5 to 0.65 would improve Precision at the cost of more spam reaching inboxes.

---

---

## Project 3 — Student Pass/Fail Prediction

**File:** `student_pass_fail.ipynb`
**Dataset:** 6,550 rows → 6,500 after cleaning
**Question:** Can we predict whether a student will pass or fail based on study habits and background?

### Dataset

| Column | Description |
|---|---|
| `study_hours_per_day` | Daily study hours |
| `attendance_percent` | Class attendance rate |
| `previous_score` | Score in previous examination |
| `sleep_hours` | Daily sleep hours |
| `assignments_done` | Number of assignments completed |
| `has_tuition` | Binary — 1 if attending tuition |
| `parent_education` | Parent education level (0, 1, 2) |
| `motivation_score` | Self-reported motivation (1–10) |
| `internet_hours` | Daily recreational internet hours |
| `passed` | **Target** — 1 = passed, 0 = failed |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `attendance_percent` | String with `_err` suffix |
| Impossible | All continuous columns | Negative values — caught in bulk loop |
| Impossible | `attendance_percent` | 150% — not explicitly caught |
| Impossible | `motivation_score` | 15 (max is 10) — not explicitly caught |
| Nulls | All columns except target | ~131 per column |
| Duplicates | All columns | 50 exact copies |

### Impossible Values — Best Approach in the Batch

```python
# All columns checked for negatives in one loop
Columns = df_copy.columns.tolist()
for col in Columns:
    df_copy.loc[df_copy[col]<0, col] = np.nan
```

Most comprehensive impossible value coverage across all four notebooks — one loop catches negatives in all columns simultaneously.

### Outlier Exclusions — Correct Logic

```python
Columns.remove('passed')        # target
Columns.remove('has_tuition')   # binary
Columns.remove('parent_education')  # categorical
```

Binary and categorical columns correctly excluded from IQR detection.

### Best Result of the Batch

```
Accuracy:   0.8754
Gap:        0.0010  ← best generalisation of all four notebooks
```

Gap of 0.0010 is essentially zero overfitting. Training and test accuracy differ by less than 0.1 percentage points. C=0.01 was the reason — the wide margin prevented the model from fitting too closely to training patterns.

### What C=0.01 Tells You

Student pass/fail is a linearly separable problem. Students who study more, attend more, and scored well previously cluster clearly into the pass group. A simple wide-margin boundary separates them from students who don't. The model doesn't need a complex, tight boundary — the data structure is simple enough that a very low C (maximum tolerance for misclassification) still achieves the best accuracy. This is a direct signal about the nature of the data.

---

---

## Project 4 — Wine Quality Classification

**File:** `wine_quality.ipynb`
**Dataset:** 5,550 rows → 5,500 after cleaning
**Question:** Can we classify wines as high or standard quality based on physicochemical measurements?

### Dataset

| Column | Description |
|---|---|
| `fixed_acidity` | Tartaric acid concentration |
| `volatile_acidity` | Acetic acid — contributes vinegar taste |
| `citric_acid` | Freshness and flavour |
| `residual_sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content |
| `free_sulfur_dioxide` | Free SO2 — prevents microbial growth |
| `total_sulfur_dioxide` | Total SO2 concentration |
| `density` | Density of wine (g/ml) |
| `ph` | Acidity/alkalinity |
| `alcohol` | Alcohol percentage |
| `high_quality` | **Target** — 1 = high quality, 0 = standard |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `fixed_acidity` | String with `_err` suffix |
| Impossible | `volatile_acidity`, `residual_sugar`, `alcohol` | Negative values |
| Impossible | `ph` | 9.0 (wine pH is always 2.8–4.5) — caught via IQR |
| Impossible | `density` | 5.0 (wine density is always ~0.99–1.004) — caught via IQR |
| Impossible | `chlorides` | 5.0 (realistic max is ~0.6) — caught via IQR |
| Nulls | All columns except target | ~110 per column |
| Duplicates | All columns | 50 exact copies |

### Outlier Approach — All Feature Columns

```python
Columns = df_copy.columns.tolist()
Columns.remove('high_quality')
```

IQR applied to all feature columns. Correct for wine chemistry data where any extreme reading is likely a measurement error — wine properties are tightly bounded by chemistry and cannot physically exceed certain ranges.

### Results

```
Accuracy:   0.7818
Gap:        0.0025  ← second best generalisation
```

Lowest accuracy of the batch. Wine quality classification is genuinely harder — the chemical differences between high and standard quality wine are subtle and don't always produce cleanly separable clusters in feature space. Two wines can have nearly identical chemistry but different sensory quality. The 0.78 accuracy reflects this real-world ambiguity rather than model weakness.

C=0.01 produced the best result here as well, consistent with the student notebook. Both datasets have relatively linear separability — the simple wide-margin SVM boundary outperforms more complex boundaries on linearly structured data.

---

---

## Results Across All Four

| Project | Accuracy | Gap | Best C | Kernel |
|---|---|---|---|---|
| Cancer Diagnosis | 0.8450 | 0.0127 | 1.0 | RBF |
| Email Spam | 0.8186 | 0.0370 | 1.0 | RBF |
| Student Pass/Fail | **0.8754** | **0.0010** | 0.01 | RBF |
| Wine Quality | 0.7818 | 0.0025 | 0.01 | RBF |

Two datasets needed C=1 (cancer, spam) — moderately complex boundaries. Two needed C=0.01 (student, wine) — simple linear-like boundaries. The C tuning loop correctly identified this in all four cases.

---

## What SVM Does Differently From Previous Models

**No feature importance** — SVM has no `feature_importances_` or `coef_` equivalent in the RBF kernel. The model's decision logic is not directly interpretable the way tree importance or logistic regression coefficients are. This is a genuine limitation for analyst work where explaining the model matters.

**Scaling is non-negotiable** — more critical here than for logistic regression. Distance calculations in SVM are directly affected by scale. A Pipeline ensures this is always done correctly without data leakage.

**C replaces max_depth** — in Decision Trees you controlled overfitting with `max_depth`. In SVM you control it with C. The principle is the same — a tuning parameter that trades off training fit against generalisation.

**Speed tradeoff** — SVM training time scales with dataset size much faster than linear. The datasets here were kept at 5,000–7,000 rows deliberately. On the 32,000-row Random Forest datasets, SVM would take significantly longer to train with minimal accuracy improvement.

---

## What's Next

- KNN — K-Nearest Neighbors, distance-based classification
- XGBoost — gradient boosting, currently one of the strongest algorithms for tabular data
- Cross-validation — more reliable model evaluation than a single train/test split
- GridSearchCV — systematic hyperparameter tuning instead of manual loops

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
