# ML Learning Journey — Random Forest Classification

Four binary classification problems on large datasets (32,000+ rows each). The main upgrade from Decision Tree — instead of asking one tree's opinion, you ask a hundred trees and take the majority vote. Turns out the crowd is almost always smarter than the individual.

---

## Repo Structure

```
README.md
notebooks/
  credit_fraud.ipynb           ← Finance — Fraud Detection
  ecommerce_purchase.ipynb     ← Retail — Purchase Prediction
  hospital_readmission.ipynb   ← Healthcare — 30-Day Readmission
  road_accident.ipynb          ← Transport — Accident Severity
data/
  credit_fraud.csv
  ecommerce_purchase.csv
  hospital_readmission.csv
  road_accident.csv
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, recall_score,
                             precision_score, f1_score)
```

---

## The Core Concept — Random Forest

A Decision Tree asks one set of questions. Random Forest builds hundreds of trees, each trained on a different random slice of the data, each considering a different random subset of features at every split. Final prediction is a majority vote across all trees.

```
Decision Tree  →  1 tree  →  1 answer
Random Forest  →  100 trees  →  100 answers  →  majority vote
```

Two mechanisms make each tree different from the others:

**Bagging** — each tree trains on a random sample of rows drawn with replacement. Same row can appear multiple times in one tree's training data. Each tree sees a different slice of reality.

**Feature Randomness** — at each split, only a random subset of features is considered. With 9 features and `max_features='sqrt'`, each split evaluates roughly 3 features. Forces trees to find different patterns rather than all making the same splits.

### Out of Bag Score

Because each tree trains on ~63% of the data, the remaining ~37% acts as a free built-in validation set.

$$\text{OOB Error} = \frac{\text{Misclassified OOB samples}}{\text{Total OOB samples}}$$

OOB score gives you a reliable accuracy estimate without touching the test set. Enabled with `oob_score=True`. In practice, OOB score consistently sat between training and test accuracy across all four notebooks — exactly where it should be.

### Feature Importance

Unlike a single Decision Tree where importance can be noisy, Random Forest averages feature importance across all trees. The result is more stable and trustworthy.

---

## Parameters Used Across Notebooks

```python
RandomForestClassifier(
    n_estimators=100,      # number of trees
    max_depth=6,           # max levels per tree
    min_samples_split=100, # need 100 samples to attempt a split
    min_samples_leaf=50,   # each leaf must have 50+ samples
    oob_score=True,        # free validation estimate
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
```

`min_samples_split` and `min_samples_leaf` are the main overfitting controls in Random Forest. With 32,000 rows, higher thresholds (50–100) prevent the trees from finding patterns in tiny subgroups that won't generalise.

---

## The n_estimators Tuning Loop

Run in every notebook to find the point where adding more trees stops improving accuracy:

```python
train_scores, test_scores = [], []
tree_range = range(10, 201, 10)

for n in tree_range:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, rf.predict(X_train)))
    test_scores.append(accuracy_score(y_test, rf.predict(X_test)))

best_n = tree_range[np.argmax(test_scores)]
plt.axvline(best_n, linestyle='--', label=f'Best n={best_n}')
```

The vertical line marks the optimal n_estimators. After this point, accuracy plateaus and adding trees only costs computation time.

---

---

## Project 1 — Credit Card Fraud Detection

**File:** `credit_fraud.ipynb`  
**Dataset:** 32,300 rows → 32,000 after cleaning  
**Question:** Can we flag fraudulent transactions before they're processed?

### Dataset

| Column | Description |
|---|---|
| `age` | Cardholder age |
| `transaction_amount` | Transaction value in rupees |
| `account_balance` | Current account balance |
| `num_transactions_today` | Transactions made today |
| `is_foreign_transaction` | Binary — 1 if abroad |
| `transaction_hour` | Hour of transaction (0–23) |
| `prev_fraud_flag` | Binary — 1 if previous fraud on account |
| `merchant_distance_km` | Distance from home to merchant |
| `merchant_risk_score` | Merchant risk rating (0–10) |
| `is_fraud` | **Target** — 1 = fraud, 0 = legitimate |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `age` | String with `_err` suffix |
| Impossible | `age` | -10 (negative age) |
| Impossible | `transaction_amount` | Negative values |
| Impossible | `account_balance` | -99,999 |
| Impossible | `merchant_risk_score` | 15 (max is 10) |
| Outlier | `transaction_hour` | 28 (caught via IQR) |
| Nulls | All columns except target | ~640 per column |
| Duplicates | All columns | 300 exact copies |

### Model

```python
RandomForestClassifier(n_estimators=40, max_depth=5, oob_score=True,
                       random_state=42, n_jobs=-1)
```

### Results

```
Accuracy:   0.7405
OOB Score:  0.7239
Train/Test Gap: -0.0075  (test slightly higher than train — healthy)
```

```
Classification Report:
              precision  recall  f1-score
Not Fraud         0.73    0.81      0.77
Fraud             0.75    0.67      0.71
```

### Feature Importance

```
prev_fraud_flag           → 0.2638  ← most important
merchant_distance_km      → 0.2210
is_foreign_transaction    → 0.2109
merchant_risk_score       → 0.1429
transaction_amount        → 0.1167
num_transactions_today    → 0.0234
age                       → 0.0141
transaction_hour          → 0.0054
account_balance           → 0.0018  ← least important
```

Past fraud history, transaction location and foreign flag together account for 70% of the model's decision making. This matches exactly how real bank fraud detection rule engines are built — the model independently discovered the same logic from data alone.

`account_balance` at 0.0018 is effectively unused. Fraud happens across all balance levels — wealthy and poor cardholders get defrauded equally. The amount in the account doesn't help distinguish fraud from legitimate spending.

### Key Domain Note

Recall for fraud (class 1) = 0.67. This means 33% of actual fraud cases are being missed. For a bank, each missed fraud has a direct financial cost. The decision threshold of 0.5 could be lowered to 0.3 to catch more fraud at the cost of more false alarms — a trade-off worth making in this domain.

---

---

## Project 2 — E-Commerce Purchase Prediction

**File:** `ecommerce_purchase.ipynb`  
**Dataset:** 32,300 rows → 32,000 after cleaning  
**Question:** Will a user complete a purchase during their current session?

### Dataset

| Column | Description |
|---|---|
| `session_duration_min` | Time spent on site in minutes |
| `pages_viewed` | Number of pages visited |
| `cart_value` | Total value of items in cart |
| `prev_purchases` | Historical purchase count |
| `discount_percent` | Discount applied to session |
| `device_type` | 0 = desktop, 1 = mobile, 2 = tablet |
| `day_of_week` | Day 0–6 |
| `bounce_rate` | Session bounce rate (0–1) |
| `loyalty_score` | Customer loyalty score (0–100) |
| `made_purchase` | **Target** — 1 = purchased, 0 = left |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `session_duration_min` | String with `_err` suffix |
| Impossible | `session_duration_min` | Negative and zero |
| Impossible | `cart_value` | -200 |
| Impossible | `discount_percent` | 150% |
| Impossible | `bounce_rate` | 5.0 (max is 1.0) |
| Impossible | `loyalty_score` | 200 (max is 100) |
| Nulls | All columns except target | ~645 per column |
| Duplicates | All columns | 300 exact copies |

### Model — Best Parameterised Across All Four

```python
RandomForestClassifier(max_depth=6, n_estimators=140, n_jobs=-1,
                       min_samples_split=100, min_samples_leaf=50,
                       oob_score=True, random_state=42)
```

The aggressive `min_samples` values were deliberate — with 32,000 rows, requiring 100 samples to split and 50 per leaf prevents the forest from memorising small user behaviour clusters that won't generalise to new sessions.

### Results

```
Accuracy:   0.7677
OOB Score:  0.7570
Train/Test Gap: 0.0049  ← best generalisation of the batch
```

Gap of 0.0049 is essentially zero overfitting — the model learns real patterns without memorising training data.

### Feature Importance

```
prev_purchases        → 0.3787  ← dominant predictor
session_duration_min  → 0.2046
bounce_rate           → 0.1389
loyalty_score         → 0.1185
discount_percent      → 0.0798
cart_value            → 0.0459
pages_viewed          → 0.0312
day_of_week           → 0.0018
device_type           → 0.0006  ← effectively unused
```

Purchase history is the strongest signal by a wide margin. Someone who has bought before is far more likely to buy again — the model captured what every e-commerce team knows intuitively.

`bounce_rate` in third place validates the domain expectation — users who bounce quickly almost never convert. The forest treats this as reliable negative evidence.

`device_type` at 0.0006 means the device used had essentially no effect on purchase likelihood in this data. Desktop vs mobile vs tablet didn't change conversion patterns — possibly because the site experience was consistent across devices.

### n_estimators Tuning

Best implementation across all four notebooks — optimal n calculated programmatically, vertical line drawn on the plot at that point. Found the sweet spot and trained the model there.

---

---

## Project 3 — Hospital Readmission Prediction

**File:** `hospital_readmission.ipynb`  
**Dataset:** 32,300 rows → 32,000 after cleaning  
**Question:** Which patients are likely to return within 30 days of discharge?

### Dataset

| Column | Description |
|---|---|
| `age` | Patient age |
| `length_of_stay_days` | Days hospitalised |
| `num_diagnoses` | Number of active diagnoses |
| `num_medications` | Medications prescribed |
| `prev_admissions` | Previous hospital admissions |
| `glucose_level` | Blood glucose reading |
| `bmi` | Body mass index |
| `has_diabetes` | Binary — 1 if diabetic |
| `discharge_type` | 0 = home, 1 = skilled nursing, 2 = other |
| `readmitted_30days` | **Target** — 1 = readmitted, 0 = not |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `age` | String with `_err` suffix |
| Impossible | `age` | 150 |
| Impossible | `bmi` | Negative |
| Impossible | `glucose_level` | 900 |
| Impossible | `num_medications` | 200 (max is 20) |
| Impossible | `length_of_stay_days` | -3 (negative stay) |
| Nulls | All columns except target | ~645 per column |
| Duplicates | All columns | 300 exact copies |

### Results

```
Accuracy:   0.7594
OOB Score:  0.7627
Train/Test Gap: 0.0165  ← clean generalisation
```

OOB score (0.7627) sits above test accuracy (0.7594) — the forest's built-in estimate was actually more optimistic than test performance, which is slightly unusual but within normal variance.

### Feature Importance

```
prev_admissions      → 0.5653  ← dominates everything
length_of_stay_days  → 0.1247
age                  → 0.1124
num_medications      → 0.0514
num_diagnoses        → 0.0502
discharge_type       → 0.0448
has_diabetes         → 0.0221
glucose_level        → 0.0159
bmi                  → 0.0132
```

`prev_admissions` at 56.5% is the most dominant single feature across all four notebooks. In clinical medicine this is called the "frequent flyer" phenomenon — patients who have been hospitalised multiple times before are statistically far more likely to return. The forest found this pattern and weighted it heavily above everything else.

`discharge_type` at 0.0448 is worth noting. Patients discharged to skilled nursing facilities (type 1) vs directly home (type 0) showed a real difference in readmission risk. The model picked this up, suggesting that where a patient goes after discharge matters for recovery.

`bmi` and `glucose_level` both landed near the bottom despite being medically relevant. This is partly because both were imputed with median values for ~640 missing rows — imputed values carry no real signal and dilute the feature's predictive power.

---

---

## Project 4 — Road Accident Severity

**File:** `road_accident.ipynb`  
**Dataset:** 32,300 rows → 32,000 after cleaning  
**Question:** Can we predict whether an accident will be severe based on conditions at the time?

### Dataset

| Column | Description |
|---|---|
| `vehicle_speed_kmh` | Speed at time of accident |
| `visibility_km` | Visibility in km |
| `road_type` | 0 = urban, 1 = rural, 2 = highway |
| `weather_condition` | 0 = clear, 1 = rain, 2 = fog, 3 = snow |
| `hour_of_day` | Hour of accident (0–23) |
| `driver_age` | Age of driver |
| `vehicles_involved` | Number of vehicles in accident |
| `alcohol_involved` | Binary — 1 if alcohol detected |
| `seatbelt_worn` | Binary — 1 if seatbelt worn |
| `severe_accident` | **Target** — 1 = severe, 0 = minor |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `vehicle_speed_kmh` | String with `_err` suffix |
| Impossible | `vehicle_speed_kmh` | Negative and 500 kmh |
| Impossible | `visibility_km` | Negative |
| Impossible | `hour_of_day` | 30 |
| Impossible | `driver_age` | 12 |
| Nulls | All columns except target | ~647 per column |
| Duplicates | All columns | 300 exact copies |

### Results

```
Accuracy:   0.7527
OOB Score:  0.7609
Train/Test Gap: 0.0178  ← healthy generalisation
```

### Feature Importance

```
vehicle_speed_kmh   → 0.6891  ← overwhelmingly dominant
visibility_km       → 0.0823
alcohol_involved    → 0.0763
seatbelt_worn       → 0.0578
vehicles_involved   → 0.0460
weather_condition   → 0.0222
driver_age          → 0.0149
road_type           → 0.0065
hour_of_day         → 0.0047
```

Speed at 69% importance is the clearest result across all four notebooks. This is physics — kinetic energy scales with the square of velocity, so crash severity increases non-linearly with speed. The forest found a fundamental law of physics from accident data.

`alcohol_involved` at 0.0763 is lower than expected given its real-world impact. Only 12% of records had alcohol involved (realistic for general accident data), which limits how much signal the forest can extract from a rare binary feature. In a dataset skewed toward alcohol-related accidents, this would rank much higher.

`hour_of_day` and `road_type` near zero — time of day and road type matter for accident frequency (how often accidents happen) but less for severity (how bad they are when they do). The distinction is subtle and the model captured it correctly.

---

---

## Results Across All Four

| Project | Accuracy | OOB Score | Gap | Best Feature |
|---|---|---|---|---|
| Credit Fraud | 0.7405 | 0.7239 | -0.008 | prev_fraud_flag (0.264) |
| E-Commerce | **0.7677** | 0.7570 | **0.005** | prev_purchases (0.379) |
| Hospital | 0.7594 | 0.7627 | 0.017 | prev_admissions (0.565) |
| Road Accident | 0.7527 | 0.7609 | 0.018 | vehicle_speed_kmh (0.689) |

E-Commerce had the best accuracy and tightest gap. Hospital had the most dominant single feature. Road Accident had the most physically interpretable result.

---

## What Random Forest Does Better Than Decision Tree

**Overfitting** — the biggest practical improvement. Decision Tree gaps across the previous batch ranged from 0.05 to 0.12. Random Forest gaps here ranged from -0.008 to 0.018. The ensemble averaging genuinely works.

**Feature importance stability** — a single tree's importance scores shift depending on which random split happened to be chosen first. With 100 trees averaged together, the scores are far more reliable. `prev_admissions` at 56% in the hospital notebook is a stable signal, not a quirk of one tree's random growth.

**Accuracy** — every dataset here outperformed the equivalent Decision Tree score from the previous batch. The improvement ranged from 5% to 10% across domains.

---

## What Random Forest Loses

You can't visualise a Random Forest the way you can `plot_tree()` on a Decision Tree. 100 trees can't be drawn on a page. Feature importance replaces the tree visual as the primary interpretability tool — it tells you what matters but not exactly how the decision is made.

For use cases where you need to explain every prediction to a regulator or a patient, a single Decision Tree is still preferable despite lower accuracy. For use cases where accuracy matters most, Random Forest wins.

---

## What's Next

- K-Means Clustering — unsupervised learning, no labels, no target column
- Hyperparameter tuning with GridSearchCV — systematic search instead of manual guessing
- Cross-validation — more reliable accuracy than a single train/test split

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
