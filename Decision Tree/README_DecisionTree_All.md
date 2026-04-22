# ML Learning Journey — Decision Tree Classification

Four binary classification problems across four completely different domains. Each notebook follows the same pipeline but applies domain-specific judgment at every cleaning step. Two notebooks include Logistic Regression as a comparison model to understand when trees outperform linear models and when they don't.

---

## Repo Structure

```
README.md
notebooks/
  machine_failure.ipynb     ← Factory sensors — Decision Tree + LR comparison
  flight_delay.ipynb        ← Aviation — Decision Tree
  customer_churn.ipynb      ← Telecom — Decision Tree
  crop_failure.ipynb        ← Agriculture — Decision Tree + LR comparison
data/
  machine_failure.csv
  flight_delay.csv
  customer_churn.csv
  crop_failure.csv
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, confusion_matrix,
                             classification_report)
```

---

## The Core Concept — Decision Tree

A Decision Tree makes predictions by asking a series of yes/no questions about the input features, working from root to leaf. Unlike Logistic Regression which draws a straight decision boundary, a tree can capture non-linear relationships and feature interactions naturally.

### How It Learns — Gini Impurity

At each node the tree evaluates every possible split on every feature and picks the one that best separates the classes. The measure used is Gini Impurity:

$$Gini = 1 - \sum_{i=1}^{k} p_i^2$$

A pure node (all one class) has Gini = 0. A perfectly mixed node has Gini = 0.5. The tree always splits toward lower Gini.

### Overfitting Control

Three parameters were used consistently across all notebooks:

```python
DecisionTreeClassifier(
    max_depth=3,           # maximum levels of questions
    min_samples_split=10,  # minimum samples needed to attempt a split
    min_samples_leaf=5,    # minimum samples required in each leaf
    random_state=42
)
```

Overfitting was checked in every notebook by comparing training vs test accuracy:

```
Gap < 0.05    → Good generalisation
Gap 0.05–0.10 → Mild overfitting
Gap > 0.10    → Serious overfitting — reduce max_depth
```

### Feature Importance

Decision Trees expose `feature_importances_` — scores between 0 and 1 summing to 1.0 — showing how much each feature contributed to Gini reduction across all splits. Unlike Logistic Regression coefficients, importance scores show magnitude only, not direction.

---

---

## Project 1 — Factory Machine Failure

**File:** `machine_failure.ipynb`  
**Models:** Decision Tree + Logistic Regression (comparison)  
**Dataset:** 612 rows → 600 after cleaning  
**Question:** Can we predict whether a machine will fail based on its age, operating conditions and maintenance history?

### Dataset

| Column | Description |
|---|---|
| `machine_age_years` | Age of machine in years |
| `operating_temp_c` | Operating temperature in Celsius |
| `vibration_hz` | Vibration frequency reading |
| `pressure_bar` | Pressure in bar |
| `rpm` | Rotations per minute |
| `days_since_maintenance` | Days since last service |
| `error_count_30days` | Errors logged in last 30 days |
| `machine_failed` | **Target** — 1 = failed, 0 = working |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `vibration_hz` | String with `_err` suffix |
| Impossible | `machine_age_years` | -4 |
| Impossible | `rpm` | -100 |
| Outlier | `operating_temp_c` | 800°C |
| Outlier | `pressure_bar` | 500 bar |
| Nulls | `operating_temp_c`, `vibration_hz` | 22 and 20 missing |
| Duplicates | All columns | 12 exact copies |

### Cleaning Decisions

`days_since_maintenance` was checked with `< 0` not `<= 0` — a maintenance day of 0 (serviced today) is valid. This kind of domain-specific threshold matters.

Outlier detection was applied only to physical sensor readings (`operating_temp_c`, `pressure_bar`, `rpm`). `error_count` and `days_since_maintenance` were intentionally excluded — extreme values in those columns are meaningful failure signals, not data errors.

### Decision Tree Results

```
Accuracy:   0.6750
F1 Score:   0.67 (class 0), 0.68 (class 1)
Train/Test Gap: 0.0688  — mild overfitting
```

### Feature Importance

```
days_since_maintenance    → 0.325  ← first question asked
error_count_30days        → 0.260
operating_temp_c          → 0.179
machine_age_years         → 0.167
pressure_bar              → 0.069
vibration_hz              → 0.000  ← never used
rpm                       → 0.000  ← never used
```

Machines not serviced for months and logging frequent errors were the strongest failure signals. The tree found this without being told — and it matches real maintenance engineering logic.

`vibration_hz` and `rpm` contributed nothing at `max_depth=3`. The other features provided enough signal that the tree never reached a split where these added value.

### Logistic Regression Comparison

```
Decision Tree:        Accuracy 0.675,  F1 0.68
Logistic Regression:  Accuracy 0.708,  F1 0.72
Winner: Logistic Regression
```

Machine failure in this dataset follows a fairly additive pattern — more risk factors present means higher failure probability. Linear models handle this kind of cumulative risk well. The Decision Tree at depth 3 didn't have enough layers to capture the interaction effects that might have given it an edge.

---

---

## Project 2 — Flight Delay Prediction

**File:** `flight_delay.ipynb`  
**Model:** Decision Tree only  
**Dataset:** 694 rows → 638 after cleaning  
**Question:** Can we predict whether a flight will be delayed based on weather conditions, route characteristics and previous flight history?

### Dataset

| Column | Description |
|---|---|
| `departure_hour` | Hour of scheduled departure (0–23) |
| `scheduled_duration_min` | Planned flight duration in minutes |
| `distance_miles` | Route distance |
| `num_connections` | Number of connecting flights |
| `wind_speed_kmh` | Wind speed at departure airport |
| `visibility_km` | Visibility in km |
| `prev_flight_delay_min` | Delay of the incoming aircraft |
| `is_delayed` | **Target** — 1 = delayed, 0 = on time |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `wind_speed_kmh` | String with `_err` suffix |
| Impossible | `departure_hour` | 28 (max is 23) |
| Impossible | `distance_miles` | 0 |
| Impossible | `prev_flight_delay_min` | Negative |
| Outlier | `wind_speed_kmh` | 500 kmh |
| Nulls | `wind_speed_kmh`, `visibility_km` | 21 and 20 missing |
| Duplicates | All columns | 14 exact copies |

### Cleaning Approach

`select_dtypes(include='number')` was used with `columns.remove('is_delayed')` to automatically loop all numeric features for outlier detection — no hardcoded column names. This is a more robust approach that scales to wider datasets.

`dropna()` was used instead of median imputation for the remaining nulls. Weather readings were treated as genuinely missing rather than imputable — a reasonable domain judgment since filling a wind speed with the median could mislead the model about actual conditions on that flight.

### Decision Tree Results

```
Accuracy:   0.6250
F1 Score:   0.64 (class 0), 0.61 (class 1)
Train/Test Gap: 0.1162  — serious overfitting threshold crossed
```

The gap of 0.1162 is above the 0.10 warning line. At `max_depth=3`, this suggests the model memorised some training-specific patterns. Reducing to `max_depth=2` or increasing `min_samples_split` to 15 would tighten this.

### Feature Importance

```
wind_speed_kmh          → 0.3321
visibility_km           → 0.2655
num_connections         → 0.1771
distance_miles          → 0.1203
prev_flight_delay_min   → 0.1050
scheduled_duration_min  → 0.0000
departure_hour          → 0.0000
```

Weather conditions dominate. Wind speed and visibility together account for 60% of the tree's decision making — which makes operational sense. Number of connections was the strongest non-weather predictor, consistent with how cascading delays work in airline networks.

`departure_hour` showing zero importance is interesting — late night departures are typically more delay-prone in real aviation, but the model didn't find this pattern at depth 3.

---

---

## Project 3 — Customer Churn Prediction

**File:** `customer_churn.ipynb`  
**Model:** Decision Tree only  
**Dataset:** 633 rows → 620 after cleaning  
**Question:** Can we predict which telecom customers will cancel their subscription based on their account and usage behaviour?

### Dataset

| Column | Description |
|---|---|
| `tenure_months` | Months as a customer |
| `monthly_charges` | Monthly bill amount |
| `num_services` | Number of active services |
| `support_calls` | Support calls made |
| `contract_type` | 0 = monthly, 1 = 1yr, 2 = 2yr |
| `avg_daily_usage_gb` | Daily data usage |
| `payment_delays` | Number of late payments |
| `churned` | **Target** — 1 = left, 0 = stayed |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `monthly_charges` | String with `_err` suffix |
| Impossible | `tenure_months` | -5 |
| Impossible | `monthly_charges` | 9999 |
| Impossible | `payment_delays` | 50 |
| Impossible | `avg_daily_usage_gb` | Negative |
| Nulls | `monthly_charges`, `support_calls` | 22 and 21 missing |
| Duplicates | All columns | 13 exact copies |

### Decision Tree Results

```
Accuracy:   0.6935
F1 Score:   0.68 (class 0), 0.70 (class 1)
Train/Test Gap: 0.0524  — just above acceptable threshold
```

Best overfitting control across all four notebooks. The gap of 0.0524 is only slightly above the 0.05 target — a small increase in `min_samples_split` would close it.

### Feature Importance

```
tenure_months       → 0.3824  ← first question asked
support_calls       → 0.2456
payment_delays      → 0.2238
num_services        → 0.1062
monthly_charges     → 0.0420
contract_type       → 0.0000  ← never used
avg_daily_usage_gb  → 0.0000  ← never used
```

Tenure being the strongest predictor makes real business sense — long-term customers are significantly less likely to churn. The tree asks first: "has this customer been with us long enough to be loyal?" Everything else follows from there.

`contract_type` showing zero importance is surprising given that in real telecom data, monthly contract customers churn far more than annual or 2-year subscribers. At `max_depth=3`, tenure, support calls and payment delays gave the tree enough signal that it never needed to look at contract type. A deeper tree would likely surface it.

---

---

## Project 4 — Crop Failure Prediction

**File:** `crop_failure.ipynb`  
**Models:** Decision Tree + Logistic Regression (comparison)  
**Dataset:** 655 rows → 435 after cleaning  
**Question:** Can we predict crop failure at the start of a growing season based on field and environmental conditions?

### Dataset

| Column | Description |
|---|---|
| `rainfall_mm` | Rainfall in millimetres |
| `temperature_c` | Average temperature in Celsius |
| `soil_ph` | Soil pH level |
| `fertilizer_kg` | Fertilizer applied in kg |
| `pesticide_use` | Binary — 1 if pesticide used |
| `sunlight_hours` | Daily sunlight hours |
| `irrigation` | Binary — 1 if irrigated |
| `crop_failed` | **Target** — 1 = failed, 0 = survived |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `soil_ph` | String with `_err` suffix |
| Impossible | `soil_ph` | Negative and above 14 |
| Impossible | `sunlight_hours` | Above 14, below 6 |
| Impossible | `fertilizer_kg` | Negative (-10) |
| Outlier | `temperature_c` | 95°C |
| Outlier | `rainfall_mm` | 5000mm |
| Nulls | `rainfall_mm`, `soil_ph` | 20 each |
| Duplicates | All columns | 15 exact copies |

### Strongest Cleaning Section Across All Four Notebooks

Domain-specific thresholds were applied correctly:
- `soil_ph` bounded between 0 and 14 — the actual chemical pH scale
- `sunlight_hours` bounded between 6 and 14 — practical growing season range

Two-step null handling was used: fill columns where median imputation is reasonable (`rainfall_mm`, `temperature_c`, `soil_ph`), then `dropna()` for anything remaining. This is the right layered approach.

### Decision Tree Results

```
Accuracy:   0.6437
F1 Score:   0.69 (class 0), 0.59 (class 1)
Train/Test Gap: 0.0862  — mild-to-serious range
```

### Feature Importance

```
soil_ph         → 0.3051
fertilizer_kg   → 0.2700
rainfall_mm     → 0.2598
irrigation      → 0.0844
sunlight_hours  → 0.0807
temperature_c   → 0.0000
pesticide_use   → 0.0000
```

The tree's first question was about soil pH. Agronomically this is correct — pH outside the 6.0–7.5 range blocks nutrient absorption regardless of how much fertilizer is applied or how well the field is irrigated. The model learned a real agricultural principle from data.

### Logistic Regression Comparison

```
Decision Tree:        Accuracy 0.644,  F1 0.64
Logistic Regression:  Accuracy 0.759,  F1 0.70
Winner: Logistic Regression
```

### Logistic Regression Coefficients — Agricultural Interpretation

```
irrigation     → -1.3543  (strongest protective factor)
pesticide_use  → -0.7946  (second strongest protection)
soil_ph        → -0.4582  (higher pH reduces failure risk within range)
sunlight_hours → -0.3465  (more sunlight = lower failure risk)
temperature_c  → +0.0807  (heat stress increases failure)
rainfall_mm    → -0.0078  (more rain helps, but small effect here)
fertilizer_kg  → -0.0101  (more fertilizer helps, modest effect)
```

All coefficients align with real crop science. Irrigation being the strongest predictor makes sense in a dataset where rainfall alone is insufficient — managed water supply is a direct control variable farmers can act on. This is the kind of coefficient reading that turns a model output into an actionable farm advisory.

---

---

## Cross-Notebook Comparison

| Project | Accuracy | F1 Score | Gap | Overfitting |
|---|---|---|---|---|
| Machine Failure | 0.675 | 0.68 | 0.069 | Mild |
| Flight Delay | 0.625 | 0.62 | 0.116 | Serious |
| Customer Churn | 0.694 | 0.69 | 0.052 | Minimal |
| Crop Failure | 0.644 | 0.64 | 0.086 | Mild-Serious |

Customer Churn had the best generalisation. Flight Delay had the most overfitting — the weather data may have had patterns in the training set that didn't fully generalise to test.

### When Did Logistic Regression Win?

Both notebooks where a comparison was run — Machine Failure and Crop Failure — Logistic Regression outperformed the Decision Tree. Both datasets had fairly linear additive risk patterns. Trees tend to win when data has sharp non-linear thresholds and interaction effects between features. Neither dataset here strongly required that.

---

## Common Patterns Across All Four

**Dynamic file paths** — `input()` used in every notebook instead of hardcoded paths. Clean habit.

**`select_dtypes` outlier loops** — two notebooks used automated column detection instead of hardcoded lists. More robust approach for wider datasets.

**Feature importance interpretation** — every notebook printed the importance table and identified which features the tree chose first. The first split choice in each tree aligned with real domain knowledge: maintenance days for machines, wind speed for flights, tenure for telecom, soil pH for crops.

**Zero-importance features** — every notebook had at least two features that contributed nothing at `max_depth=3`. This is expected — shallow trees don't always need all features to find good splits. A deeper tree or more data might activate them.

---

## What's Next

- Random Forest — ensemble of Decision Trees, generally stronger and more stable
- Hyperparameter tuning — systematically finding the best `max_depth` instead of guessing
- StandardScaler Pipeline for Logistic Regression — fix the coefficient scaling issues found in Machine Failure
- Cross-validation — more reliable accuracy estimate than a single train/test split
- K-Means Clustering — unsupervised grouping without labels

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
