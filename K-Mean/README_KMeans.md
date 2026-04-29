# ML Learning Journey — K-Means Clustering

Four unsupervised learning problems across four completely different domains. No target column. No right answer. No accuracy score. The goal shifts from prediction to discovery — finding hidden structure in data that nobody labelled in advance.

---

## Repo Structure

```
README.md
notebooks/
  air_quality.ipynb           ← Environment — Pollution Zone Classification
  cricket_players.ipynb       ← Sports — Player Role Discovery
  customer_spending.ipynb     ← Retail — Customer Segment Discovery
  employee_performance.ipynb  ← HR — Workforce Tier Identification
data/
  air_quality.csv
  cricket_players.csv
  customer_spending.csv
  employee_performance.csv
```

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import floor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
```

---

## What Changed From Previous Models

Everything before this had a target column. K-Means does not.

```
Supervised Learning    →  Data + Labels  →  Model learns rules  →  Predicts known outcomes
Unsupervised Learning  →  Data only      →  Model finds groups  →  Discovers unknown structure
```

Three structural changes in every notebook:

```python
# 1. No y — no target column
X = df_clean[all_columns]        # just features

# 2. No train/test split
# K-Means uses all the data — splitting makes no sense here

# 3. Scaling is mandatory
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Scaling is not optional in K-Means. Distance calculations treat larger-scale features as more important. Without StandardScaler, a column with values in the thousands dominates clustering over a column with values in the tens, regardless of actual relevance.

---

## The Algorithm

```
Step 1  →  Choose K (number of clusters)
Step 2  →  Randomly place K centroids in feature space
Step 3  →  Assign every row to its nearest centroid
Step 4  →  Move each centroid to the mean of its assigned rows
Step 5  →  Repeat steps 3 and 4 until centroids stop moving
```

Similarity is measured with Euclidean distance:

$$d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + ... + (z_1 - z_2)^2}$$

---

## Finding Optimal K — Two Methods Used Together

### Elbow Method — Inertia vs K

$$\text{Inertia} = \sum_{i=1}^{n} \min_{\mu_j \in C} ||x_i - \mu_j||^2$$

Total distance of all points from their cluster centroid. Lower is better. The curve bends at the optimal K — adding more clusters past that point gives diminishing returns.

### Silhouette Score

$$s = \frac{b - a}{\max(a, b)}$$

Where `a` = average distance to own cluster members, `b` = average distance to nearest other cluster. Score ranges from -1 to 1. Higher is better.

Both methods were always run together and plotted side by side. When they disagreed, domain knowledge was used to make the final call.

### K Range Design Choice

Three of four notebooks used `range(3, 12, 2)` — testing only odd values of K. The reasoning: even K tends to produce symmetric cluster pairs that can mask genuine structure by splitting one natural group into two mirrored halves. Odd K forces the algorithm to find asymmetric groupings that better reflect real-world variation.

---

## The Full Pipeline Used Across All Notebooks

```
Load Data
    ↓
Explore (shape, info, describe, nulls, duplicates)
    ↓
Copy — never modify original
    ↓
Fix Data Types (regex + astype)
    ↓
Impossible Values → nullify
    ↓
Outliers (IQR on selected columns) → nullify
    ↓
Drop Duplicates
    ↓
Fill Nulls (floor for integers, round for continuous)
    ↓
Define X (all columns — no target)
    ↓
StandardScaler
    ↓
Find Optimal K (Elbow + Silhouette, both plotted)
    ↓
Train KMeans(n_clusters=K, n_init=10, random_state=42)
    ↓
Assign cluster labels → df_copy['cluster'] = model.labels_
    ↓
Cluster profile table → groupby('cluster').mean()
    ↓
PCA scatter plot (2D visualisation)
    ↓
% Difference from average chart (per cluster)
```

---

---

## Project 1 — Air Quality Classification

**File:** `air_quality.ipynb`  
**Dataset:** 30,300 rows → 30,003 after cleaning  
**Question:** What distinct pollution profiles exist across monitoring stations?

### Dataset

| Column | Description |
|---|---|
| `pm25` | Fine particulate matter (μg/m³) |
| `pm10` | Coarse particulate matter (μg/m³) |
| `no2_ppb` | Nitrogen dioxide in parts per billion |
| `co_ppm` | Carbon monoxide in parts per million |
| `so2_ppb` | Sulfur dioxide in parts per billion |
| `o3_ppb` | Ozone in parts per billion |
| `wind_speed_kmh` | Wind speed |
| `humidity_pct` | Relative humidity |
| `temperature_c` | Temperature in Celsius |
| `aqi` | Air Quality Index (composite score) |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `pm25` | String with `_err` suffix |
| Impossible | `no2_ppb`, `co_ppm`, `so2_ppb`, `pm25` | Negative readings |
| Impossible | `humidity_pct` | Above 100% |
| Outlier | `pm25`, `pm10`, `no2_ppb`, `aqi`, `co_ppm`, `o3_ppb` | Sensor spikes |
| Nulls | All columns | ~600 per column |
| Duplicates | All columns | 297 exact copies |

Outlier detection was applied selectively — only pollutant sensor columns, not meteorological readings like wind speed, humidity and temperature. These weather variables don't have sensor spikes the same way air quality instruments do.

### Optimal K

Chose K=5 from odd-K range (3, 5, 7, 9, 11). Elbow showed a clear bend at 5. Silhouette confirmed.

### Cluster Profiles

| Cluster | pm25 | AQI | Wind Speed | Profile |
|---|---|---|---|---|
| 0 | 45 | 115 | 10.0 | Moderate Pollution |
| 1 | 148 | 277 | 6.0 | Industrial Heavy |
| 2 | 84 | 194 | 4.1 | Traffic Hotspot (NO2=119) |
| 3 | 13 | 44 | 20.6 | Clean Zone — High Wind |
| 4 | 13 | 44 | 14.5 | Clean Zone — Lower Wind |

Clusters 3 and 4 are nearly identical in pollution levels but differ in wind speed. The model split clean zones into two sub-groups based on wind dispersal patterns — a finding with real environmental policy implications. High-wind clean zones remain clean passively. Lower-wind clean zones may need active monitoring since they're one windless day away from pollutant accumulation.

Cluster 2 identifies Traffic Hotspots specifically through NO2 (119 ppb average) — nitrogen dioxide is produced primarily by vehicle exhaust, making this a transport-driven pollution profile distinct from industrial sources in Cluster 1.

---

---

## Project 2 — Cricket Player Role Discovery

**File:** `cricket_players.ipynb`  
**Dataset:** 30,300 rows → 30,000 after cleaning  
**Question:** What distinct player types exist based purely on statistics?

### Dataset

| Column | Description |
|---|---|
| `batting_avg` | Runs per dismissal |
| `strike_rate` | Runs per 100 balls faced |
| `centuries` | Hundreds scored |
| `half_centuries` | Fifties scored |
| `fours_per_innings` | Average boundaries per innings |
| `sixes_per_innings` | Average maximums per innings |
| `bowling_avg` | Runs conceded per wicket |
| `economy_rate` | Runs conceded per over |
| `wickets` | Total wickets taken |
| `fielding_dismissals` | Catches, stumpings, run-outs |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `batting_avg` | String with `_err` suffix |
| Impossible | 8 columns | Negative cricket statistics |
| Outlier | `strike_rate` | 500 (impossible — caught via IQR) |
| Nulls | All columns | ~600 per column |
| Duplicates | All columns | 300 exact copies |

IQR was applied to all columns — valid since every cricket statistic is numerical and extreme values are errors. Acknowledged trade-off: applying IQR to `centuries` flags all-time great batsmen as outliers, but in a 30,000-row synthetic dataset this was acceptable.

### Cluster Profiles

| Cluster | Batting Avg | Strike Rate | Bowling Avg | Wickets | Profile |
|---|---|---|---|---|---|
| 0 | 38 | 132 | 30 | ~40 | All Rounder |
| 1 | 15 | 86 | 22 | ~80 | Bowler Specialist |
| 2 | 32 | 155 | 55 | ~8 | Power Hitter |
| 3 | 52 | 118 | 70 | ~3 | Anchor Batsman |
| 4 | 38 | 132 | 30 | ~40 | All Rounder (variant) |

K-Means independently recovered all four classic cricket player archetypes from raw statistics — without being given any positional labels. The algorithm discovered what coaches and selectors know intuitively: players cluster naturally into batsmen, bowlers, all-rounders and power hitters based on their numbers alone.

Clusters 0 and 4 are nearly identical All Rounders — suggesting K=5 over-split this group. Running K=4 would likely merge them into a single cohesive all-rounder cluster. This is a case where the data supports K=4 despite the elbow suggesting 5.

The sharpest contrast: Cluster 1 (Bowler Specialist) has batting_avg=15 and wickets=80. Cluster 3 (Anchor Batsman) has batting_avg=52 and wickets=3. These two groups are as different as it gets in cricket — the model separated them correctly with no guidance.

---

---

## Project 3 — Customer Spending Segmentation

**File:** `customer_spending.ipynb`  
**Dataset:** 32,300 rows → 32,001 after cleaning  
**Question:** What distinct customer types exist based on spending and engagement behaviour?

### Dataset

| Column | Description |
|---|---|
| `annual_income` | Yearly income in rupees |
| `monthly_spend` | Monthly spending at the store |
| `avg_basket_size` | Average transaction value |
| `purchase_frequency` | Purchases per month |
| `discount_usage_pct` | % of purchases made with discount |
| `loyalty_points` | Accumulated loyalty points |
| `online_orders_pct` | % of orders placed online |
| `return_rate_pct` | % of items returned |
| `customer_age` | Age of customer |
| `months_active` | Account tenure in months |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `annual_income` | String with `_err` suffix |
| Impossible | `annual_income`, `monthly_spend`, `return_rate_pct`, `customer_age` | Negative values |
| Impossible | `customer_age` | Above 100 |
| Impossible | `discount_usage_pct` | 150% |
| Outlier | `annual_income`, `monthly_spend`, `loyalty_points` | Extreme high values |
| Nulls | All columns | ~640 per column |
| Duplicates | All columns | 299 exact copies |

Best imputation approach across all four notebooks — `floor()` for `customer_age` (whole number, fractions don't represent real ages), `round()` for all continuous financial variables. The deliberate split reflects domain awareness about what each column actually represents.

### Optimal K

This was the only notebook where K_range used `range(2, 11)` — all values tested rather than only odd. More thorough, slightly inconsistent with the other three notebooks, but not incorrect.

### Cluster Profiles

| Cluster | Income | Monthly Spend | Discount % | Loyalty Points | Profile |
|---|---|---|---|---|---|
| 0 | 1.8L | 2,055 | 74% | 825 | Budget Shoppers |
| 1 | 4.5L | 7,942 | 40% | 3,478 | Regular Buyers |
| 2 | 2.6L | 4,803 | 11% | 2,139 | Premium Basket (high basket size) |
| 3 | 2.5L | 4,505 | 89% | 1,997 | Bargain Hunters |
| 4 | 9.1L | 4,940 | 11% | 2,148 | Ultra-Premium (464 rows only) |

Cluster 4 contains only 464 customers — 1.4% of the dataset. This micro-cluster represents extremely high income individuals (₹9.1L/month) with near-zero discount usage. They spend moderately relative to income, buy at full price, and are the most valuable customers by margin. Small cluster size is expected for this segment — genuinely wealthy buyers are rare.

Cluster 3 is the Bargain Hunter profile — 89% discount usage, highest purchase frequency, but lower average basket size. These customers are active and engaged but primarily when promotions are running. Marketing to this segment should focus on loyalty programmes rather than deeper discounts.

The contrast between Budget Shoppers (Cluster 0) and Bargain Hunters (Cluster 3) is instructive — both use high discounts, but Budget Shoppers have genuinely low income while Bargain Hunters have moderate income and choose to wait for deals. Same behaviour, different underlying motivation.

---

---

## Project 4 — Employee Performance Segmentation

**File:** `employee_performance.ipynb`  
**Dataset:** 31,300 rows → 31,001 after cleaning  
**Question:** What distinct workforce segments exist based on performance and engagement?

### Dataset

| Column | Description |
|---|---|
| `performance_score` | Overall performance rating (0–100) |
| `projects_completed` | Projects delivered in the period |
| `weekly_hours` | Average hours worked per week |
| `training_hours_yr` | Training hours completed annually |
| `salary_lakh` | Annual salary in lakhs |
| `promotions` | Number of promotions received |
| `attendance_pct` | Attendance percentage |
| `satisfaction_score` | Employee satisfaction rating (1–5) |
| `years_experience` | Total work experience |
| `team_rating` | Rating given by team members |

### Dirty Data

| Problem | Column | Value |
|---|---|---|
| Wrong type | `performance_score` | String with `_err` suffix |
| Impossible | `performance_score` | Above 100 |
| Impossible | `projects_completed`, `training_hours_yr`, `salary_lakh`, `attendance_pct` | Negative values |
| Outlier | `weekly_hours` | 200 (more than hours in a week) |
| Outlier | `training_hours_yr`, `salary_lakh`, `years_experience` | IQR flagged extremes |
| Nulls | All columns | ~625 per column |
| Duplicates | All columns | 299 exact copies |

`salary_lakh` accumulated 1,110 nulls after cleaning — the most of any column. IQR removed a wide band of salary outliers on top of the impossible negative values. As a result, 3.5% of salary values were imputed with the median. This compresses salary variation slightly — a known trade-off when cleaning heavily dirty financial columns.

### Cluster Profiles

| Cluster | Performance | Weekly Hours | Satisfaction | Training Hours | Profile |
|---|---|---|---|---|---|
| 0 | 48 | 36 | 1.8 | 8.5 | Disengaged |
| 1 | 72 | 42 | 3.5 | 39.7 | Steady Worker |
| 2 | 92 | 48 | 4.2 | 78.4 | High Performer |
| 3 | 65 | 62 | 2.5 | 20.1 | Burnout Risk |
| 4 | 72 | 42 | 3.5 | 39.6 | Steady Worker (variant) |

Cluster 3 is the most urgent finding in this dataset. These employees work the longest hours (62 per week) but deliver below-average performance (65) with the second-lowest satisfaction score (2.5) and minimal training investment. This is the textbook burnout profile — overworked, under-supported, declining output.

The gap between Cluster 2 (High Performers) and Cluster 3 (Burnout Risk) is stark. High Performers work 48 hours with high satisfaction and heavy training investment. Burnout employees work 62 hours with low satisfaction and almost no training. The extra 14 hours of work per week is producing worse outcomes — exactly what burnout research predicts.

Clusters 1 and 4 are near-identical Steady Workers — suggesting K=4 may be a better fit for this data. At K=4, these two groups would merge into a single cohesive mid-performer cluster, giving cleaner separation between the four natural groups (Disengaged, Steady, High Performer, Burnout).

---

---

## Findings Across All Four Notebooks

### The K=5 Pattern

All four notebooks chose K=5. In three of them (Air Quality, Cricket, Employee), K=5 produced two nearly identical clusters that represented the same natural group split by minor variation. This is a consistent signal that K=4 may actually be the true number of natural segments in each dataset. The odd-K range approach helped surface this — by skipping K=4 in three notebooks, the model was pushed to K=5 which over-segmented slightly.

### Scaling Confirmed Its Importance

Every notebook showed dramatic differences in raw column ranges before scaling — AQI vs temperature, annual income vs purchase frequency, batting average vs wickets. After StandardScaler, every column had mean=0 and std=1 confirmed in the describe output. The cluster profiles were interpretable because of this — without scaling the results would have been dominated by high-scale columns.

### Cluster Profiles Over Accuracy

There is no accuracy metric in K-Means. The quality of the work lives entirely in the cluster profile table and what it reveals. The cricket notebook found the four cricket archetypes. The employee notebook found the burnout cohort. The air quality notebook found the wind-based subdivision of clean zones. These findings are the output — not a number.

---

## What's Next

- Hyperparameter tuning with GridSearchCV — systematic search instead of manual guessing
- Cross-validation — more reliable evaluation than a single train/test split
- Feature Engineering — creating new columns that improve model performance
- Combining supervised and unsupervised — use K-Means cluster labels as features in a classification model

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
