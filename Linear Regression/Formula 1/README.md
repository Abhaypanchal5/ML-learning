# ML Learning Journey — Linear Regression on F1 Lap Data

This repo documents my learning path from data analyst to machine learning practitioner. I used Formula 1 lap data to make the concepts stick — because abstract examples don't cut it when you're trying to actually understand what's happening under the hood.

Everything here is real work: the mistakes, the fixes, the outputs. Not a cleaned-up tutorial.

---

## What's In This Repo

```
README.md                  ← you are here
f1_raw.csv                 ← synthetic F1 dataset with intentional dirty data
linear_regression.ipynb    ← full notebook with all tasks
charts/
  actual_vs_predicted.png  ← model evaluation chart
  tyre_age_regression.png  ← regression line visualisation
```

---

## The Problem Statement

> Can we predict a driver's lap time based on tyre age, fuel load, track temperature, and starting grid position?

Simple question. Turns out a well-cleaned dataset and four features can get you a 92% accurate model. The hard part wasn't the ML — it was getting the data ready for it.

---

## Libraries Used

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
```

---

## Stage 1 — Understanding ML Before Writing Code

Before touching sklearn, I spent time getting the mental model right. Three concepts that everything else depends on.

### How a Model Actually Learns

Traditional programming gives the machine rules and data, and it produces answers. ML flips that — you give it data and answers, and it figures out the rules itself.

The learning loop works like this:

```
1. Model makes a prediction (random guess initially)
2. Compare prediction to actual answer
3. Calculate how wrong it was  →  this is called LOSS
4. Adjust internal weights slightly to reduce loss
5. Repeat thousands of times until error is small enough
```

Those internal numbers being adjusted are called **weights** or **parameters**. Training is just finding the best set of numbers.

### Train / Test Split

The rule that matters most. You never test a model on data it trained on — that's like giving a student the exam paper during revision. They'll score 100% and learn nothing.

```
Your Dataset (100%)
        ↓
   Split before anything else
        ↓
Training Set  80%  →  model learns from this
Test Set      20%  →  model never sees this until final evaluation
```

The test set is untouchable until the very end.

### Overfitting vs Underfitting

The central tension in all of ML.

```
UNDERFITTING
Model too simple. Didn't learn enough.
Bad on training data. Bad on test data.

         ↓

    JUST RIGHT  ←  the goal

         ↓

OVERFITTING
Model memorised training data.
Great on training data. Falls apart on new data.
```

How to spot it:

| Situation | Train Accuracy | Test Accuracy |
|---|---|---|
| Underfitting | Low | Low |
| Just Right | High | High |
| Overfitting | Very High | Low |

The gap between training and test accuracy is the warning signal.

---

## Stage 2 — The Dataset

I used a synthetic F1 lap dataset with 510 rows and 7 columns. The dataset was deliberately built with real-world data problems baked in — because clean tutorial data doesn't prepare you for actual work.

| Column | Description |
|---|---|
| `driver` | Driver name |
| `tyre_compound` | Soft / Medium / Hard |
| `tyre_age` | Laps completed on current tyre |
| `fuel_load` | Fuel in kg at lap start |
| `track_temp` | Track surface temperature in °C |
| `starting_position` | Grid position at race start |
| `lap_time` | Lap time in seconds — **target variable** |

### Problems Found in Raw Data

Running `df.info()`, `df.describe()`, and `df.isnull().sum()` revealed:

| Problem | Column Affected | What It Looked Like |
|---|---|---|
| Wrong data type | `tyre_age`, `lap_time` | Stored as `object` (string) instead of float |
| Null values | `tyre_age`, `fuel_load` | 30 missing values total |
| Duplicate rows | All columns | 10 exact duplicate rows |
| Outliers | `lap_time` | Values of 245.5 and -12.3 seconds |
| Outlier | `tyre_age` | Value of 180 laps (physically impossible) |
| Impossible value | `starting_position` | -5 (can't be negative) |
| Impossible value | `fuel_load` | 300kg (F1 max is 110kg) |
| Inconsistent categories | `tyre_compound` | 'soft', 'MEDIUM', 'Hard' all in same column |
| Inconsistent categories | `driver` | 'hamilton' vs 'Hamilton' |

---

## Stage 3 — Data Cleaning

### Rule Zero

```python
df_clean = df.copy()  # never touch the original
```

### Fix 1 — Data Types

```python
df_clean['tyre_age']  = pd.to_numeric(df_clean['tyre_age'],  errors='coerce')
df_clean['lap_time']  = pd.to_numeric(df_clean['lap_time'],  errors='coerce')
```

`errors='coerce'` converts anything unreadable into `NaN` instead of crashing. Safe default for type conversion.

### Fix 2 — Inconsistent Categories

```python
df_clean['tyre_compound'] = df_clean['tyre_compound'].str.strip().str.title()
df_clean['driver']        = df_clean['driver'].str.strip().str.title()
```

`.str.title()` capitalises the first letter of every word. One line fixes every casing variation.

### Fix 3 — Impossible Values

```python
df_clean.loc[df_clean['starting_position'] < 1,  'starting_position'] = np.nan
df_clean.loc[df_clean['starting_position'] > 20, 'starting_position'] = np.nan
df_clean.loc[df_clean['fuel_load'] > 110,         'fuel_load']         = np.nan
```

Convert to `NaN` rather than dropping — handled in bulk with nulls next.

### Fix 4 — Outliers (IQR Method)

```python
Q1  = df_clean['lap_time'].quantile(0.25)
Q3  = df_clean['lap_time'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean.loc[df_clean['lap_time'] < lower, 'lap_time'] = np.nan
df_clean.loc[df_clean['lap_time'] > upper, 'lap_time'] = np.nan
```

IQR (Interquartile Range) defines the middle 50% of your data. Anything sitting 1.5× beyond that boundary is flagged as an outlier. Not deleted immediately — nullified and handled with the rest of the missing values.

### Fix 5 — Duplicates

```python
df_clean.drop_duplicates(inplace=True)
```

### Fix 6 — Null Values

```python
df_clean['tyre_age']          = df_clean['tyre_age'].fillna(df_clean['tyre_age'].median())
df_clean['fuel_load']         = df_clean['fuel_load'].fillna(df_clean['fuel_load'].median())
df_clean['lap_time']          = df_clean['lap_time'].fillna(df_clean['lap_time'].median())
df_clean['starting_position'] = df_clean['starting_position'].fillna(df_clean['starting_position'].median())
```

Median over mean because median isn't pulled by the outliers we just nullified. More stable imputation.

---

## Stage 4 — Linear Regression

### What It Is

Find the best straight line through the data that predicts a continuous number. Given tyre age, fuel load, track temp, and grid position — predict lap time.

### The Maths

Single feature version (the line you know from school):

$$y = \beta_0 + \beta_1 x$$

Multiple features version (what we actually used):

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3 + \beta_4 x_4$$

Where:
- $\hat{y}$ = predicted lap time
- $\beta_0$ = intercept (baseline lap time when all features are zero)
- $\beta_1, \beta_2...$ = coefficients (how much each feature shifts the prediction)
- $x_1, x_2...$ = feature values (tyre age, fuel load, track temp, grid position)

### How It Finds the Best Line — OLS

The model tries to minimise the **Sum of Squared Errors (SSE)**:

$$SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Why squared? Two reasons. Squaring removes negatives so positive and negative errors don't cancel out. It also penalises larger errors more heavily than small ones.

The model adjusts $\beta$ values until SSE is as small as possible. That's Ordinary Least Squares (OLS) — the engine behind `LinearRegression()`.

### Building the Model

```python
# Define features and target
X = df_clean[['tyre_age', 'fuel_load', 'track_temp', 'starting_position']]
y = df_clean['lap_time']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
```

### What the Model Learned

```
Intercept (β₀):    87.8368

tyre_age          →  +0.3188
fuel_load         →  +0.0790
track_temp        →  -0.0530
starting_position →  +0.0293
```

Reading these coefficients:

**tyre_age → +0.3188**
Every lap added to tyre age increases lap time by 0.32 seconds. A 30-lap-old tyre is about 9.6 seconds slower than a fresh one. Tyre degradation is the biggest performance factor in the model — exactly what F1 race strategy is built around.

**fuel_load → +0.0790**
Every extra kg of fuel adds 0.08 seconds. A full 110kg tank vs near-empty is roughly 8.7 seconds per lap difference. Cars get faster naturally as fuel burns off across a race.

**track_temp → -0.0530**
Negative coefficient. Hotter track = faster lap times. Warmer tarmac improves tyre grip and rubber activation. 10°C warmer track = about 0.5 seconds per lap faster.

**starting_position → +0.0293**
Smallest effect. Further back on the grid = marginally slower lap times. The impact is small because grid position matters most at lap 1 — the field spreads out quickly after that.

Full model formula:

$$\text{Lap Time} = 87.84 + (0.32 \times \text{tyre\_age}) + (0.08 \times \text{fuel\_load}) + (-0.05 \times \text{track\_temp}) + (0.03 \times \text{grid\_pos})$$

---

## Stage 5 — Evaluation

### Metrics

**MAE — Mean Absolute Error**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

Average of how far off each prediction is. In seconds.

**RMSE — Root Mean Squared Error**

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

Like MAE but large errors are punished more heavily. Always higher than MAE. The gap between them tells you whether errors are consistent or whether a few big mistakes are skewing things.

**R² Score**

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

Compares model error to the error you'd get by just predicting the average every time. R² = 1.0 is perfect. R² = 0 means the model is no better than guessing the mean.

### Results

```
MAE:       0.7309 seconds
RMSE:      1.1223 seconds
R² Score:  0.9212
```

The model explains 92.12% of lap time variation using just four features. On average predictions are 0.73 seconds from the actual value. The small gap between MAE and RMSE (0.39) means errors are fairly consistent — no single bad prediction pulling everything off.

---

## Stage 6 — Visualisations

### Chart 1 — Actual vs Predicted Lap Times

![Actual vs Predicted](charts/actual_vs_predicted.png)

Each dot is one lap from the test set. X-axis is the actual lap time, Y-axis is what the model predicted. The red dashed line is where a perfect model would sit — predicted equals actual exactly.

Dots hugging the line across the full range (88 to 107 seconds) means the model performs consistently for both fast and slow laps. No collapse in accuracy at the extremes.

### Chart 2 — Tyre Age vs Lap Time with Regression Line

![Tyre Age Regression](charts/tyre_age_regression.png)

Blue dots are actual laps. The red line is what the model predicts as tyre age increases, with all other features held at their median values.

The line rises steadily left to right — older tyres, slower laps. The vertical spread of dots around the line at any given tyre age comes from the other three features (fuel load, track temp, grid position) varying independently.

The line was built using `np.linspace()` to generate 300 evenly spaced tyre age values, with other features frozen at median. This isolates tyre age's effect cleanly.

---

## Problems Faced and How They Were Fixed

**Problem 1 — Jagged regression line on Chart 2**

First attempt at plotting the regression line connected test set predictions in data order rather than sorted by tyre age. The line zigzagged all over the chart.

Fix: Generated a smooth range of 300 tyre age values using `np.linspace()`, froze other features at median, predicted on that clean input, then plotted.

**Problem 2 — Type conversion crashing**

Trying to convert `tyre_age` directly with `astype(float)` threw an error because some cells had been stored as non-numeric strings during the dirty data injection.

Fix: Used `pd.to_numeric(errors='coerce')` which handles bad values gracefully by converting them to NaN instead of stopping execution.

**Problem 3 — Outlier removal affecting null imputation**

Nullifying outliers before imputation meant the median was calculated on already-cleaned data, which gave a more accurate fill value. Doing it the other way around (imputing first, then removing outliers) would have introduced imputed values that were themselves based on dirty data.

Fix: Always null out impossible values and outliers before running any imputation step.

---

## Key Takeaways

The maths behind Linear Regression — OLS minimising squared errors — is straightforward once you see it as a search problem. The model is searching for the weights that make predictions as close to reality as possible.

Feature coefficients tell a story. +0.32 on tyre age isn't just a number — it's the rate of tyre degradation per lap, extracted automatically from data. That's what makes ML useful for analysis, not just prediction.

Data cleaning took longer than model building. That ratio holds in real work too.

R² of 0.92 on a synthetic dataset will be lower on real F1 data — real races have safety cars, mechanical failures, strategic variation, and driver differences that four sensor readings can't fully capture. 0.65 to 0.75 on real data would be a solid result.

---

## What's Next

- **Logistic Regression** — predicting whether a driver finishes in the points (yes/no)
- **Decision Tree** — more flexible, non-linear relationships
- **Random Forest** — ensemble method, generally stronger than a single tree
- **K-Means Clustering** — grouping drivers by performance profile without labels
- **Feature Engineering** — encoding tyre compound, adding interaction terms

---

## Stack

Python 3 · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · Jupyter Notebook
