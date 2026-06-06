# Traffic Demand Prediction
 
**Author:** Pranshu Gupta
**Metric:** `score = max(0, 100 × R²(actual, predicted))`
**Target column:** `demand` | **Submission shape:** 41,778 × 2
 
---
 
## Problem Overview
 
Predict traffic demand — a continuous value bounded in `(0, 1]` — at a given geographic location and timestamp, using road, weather, and spatial features.
 
**Input features:**
 
| Feature | Description |
|---|---|
| `geohash` | Geographic location identifier |
| `day` | Day of the week |
| `timestamp` | Time of the record |
| `RoadType` | Type of nearby road |
| `NumberofLanes` | Number of lanes at the location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks exist nearby |
| `Temperature` | Temperature at the location |
| `Weather` | Weather condition |
 
---
 
## Solution Pipeline
 
```
Data Distribution (log-normal)
        ↓
Loss = MSE on log(y) + Regularization
        ↓
Optimization → RF (bootstrap averaging) + GBM (gradient descent)
        ↓
Parameters Θ* → learned embeddings, tree splits, leaf values
        ↓
Predictions ŷ_log → demand = exp(ŷ_log)
        ↓
R² Score on original demand scale → maximize
```
 
Every step is a direct consequence of the one before it. The data distribution dictates the loss; the loss dictates the optimizer; the optimizer finds Θ\*; Θ\* generates predictions; predictions are evaluated by R².
 
---
 
## Step 1 — Parameter Identification
 
Before building any model, all learnable parameters Θ are enumerated upfront.
 
### Group A — Embedding Parameters
 
| Embedding | Shape | Parameters |
|---|---|---|
| `E_geohash` | ℝ^(1249 × 8) | 9,992 |
| `E_roadtype` | ℝ^(3 × 3) | 9 |
| `E_weather` | ℝ^(4 × 2) | 8 |
 
These capture geographic and contextual identity in a continuous representation space.
 
### Group B — Cyclic Temporal Encodings (fixed, not learned)
 
```
hour_sin = sin(2π·hour/24),   hour_cos = cos(2π·hour/24)
min_sin  = sin(2π·min/60),    min_cos  = cos(2π·min/60)
day_sin  = sin(2π·day/7),     day_cos  = cos(2π·day/7)
```
 
6 deterministic transforms, 0 learnable parameters. Preserves the cyclic nature of time (23:59 is close to 00:00).
 
### Group C — Dense Layer Parameters
 
Architecture: `[23 → 128 → 64 → 32 → 1]`
 
```
W₁ ∈ ℝ^(23×128),  W₂ ∈ ℝ^(128×64)
W₃ ∈ ℝ^(64×32),   W₄ ∈ ℝ^(32×1)
Total dense params: 13,441
```
 
### Group D — Regularization Hyperparameters
 
- `λ₁` — L2 weight decay (controls model complexity)
- `λ₂` — L1 sparsity coefficient (promotes sparse weights)
- `δ` — Huber loss threshold (boundary between L1 & L2 loss regions)
**Total learnable parameters |Θ| ≈ 23,450**
 
> Knowing |Θ| upfront tells us whether we risk underfitting (too few params) or overfitting (too many params relative to 77,299 training samples).
 
---
 
## Step 2 — Target Distribution Analysis
 
Before choosing any loss function, the distribution of `demand` was analysed empirically.
 
| Statistic | Value |
|---|---|
| Min | ≈ 0.000000 |
| Max | ≈ 1.000 |
| Skewness | > 0 (right-skewed) |
| Kurtosis | > 0 (heavier tails than Gaussian) |
| Normality test (D'Agostino) | **REJECTED** (p << 0.05) |
| log(demand) skewness | ≈ 0 after transformation |
 
**Interpretation:** Demand is always positive, and multiplicative effects (weather × road type × time of day) govern it — a hallmark of log-normal data. Predicting `log(demand)` instead of `demand` directly stabilizes variance and linearizes the problem.
 
**Distribution decision:**
 
```
demand ~ LogNormal(μ(x,Θ), σ²)
Predict:  ŷ = log(demand)
Recover:  demand = exp(ŷ)
```
 
---
 
## Step 3 — Loss Function Derivation
 
The loss function is derived from the distribution assumption using Maximum Likelihood Estimation (MLE).
 
If `demand ~ LogNormal`, then:
 
```
-log P(y | x, Θ) = MSE applied to log(y)
```
 
**Full loss (MAP perspective with regularization):**
 
```
L_total(Θ) = L_data(Θ) + L_regularization(Θ)
 
L_data(Θ)           = (1/n) Σ [log(yᵢ) − ŷᵢ]²      ← MSE in log space (MLE)
L_regularization(Θ) = λ₁·‖Θ‖²  +  λ₂·‖Θ‖₁          ← L2 + L1 on weights
```
 
| Interpretation | Meaning |
|---|---|
| MLE | Minimize `L_data(Θ)` — no prior on Θ |
| MAP | Minimize `L_data(Θ) + L_reg(Θ)` — prior on Θ |
| L2 regularization | Gaussian prior on Θ |
| L1 regularization | Laplacian prior on Θ |
 
> Huber loss was also considered as an alternative when heavy-tailed residuals were observed in validation.
 
---
 
## Step 4 — Feature Engineering
 
### Temporal Features
 
| Feature | Description |
|---|---|
| `hour_sin`, `hour_cos` | Cyclic hour encoding |
| `min_sin`, `min_cos` | Cyclic minute encoding |
| `day_sin`, `day_cos` | Cyclic day-of-week encoding |
| `time_minutes` | Total minutes since midnight |
| `peak_hour` | Binary flag: 1 if hour ∈ [8, 12], else 0 |
 
### Geographic Features
 
| Feature | Description |
|---|---|
| `geo_mean_demand` | Mean demand per full geohash (target encoding) |
| `geo_std_demand` | Std deviation of demand per geohash |
| `geo_l4_mean` | Mean demand by 4-char geohash prefix (broad region) |
| `geo_l5_mean` | Mean demand by 5-char geohash prefix (mid-level region) |
 
Hierarchical geo-features give the model spatial context at multiple resolutions.
 
### Categorical & Other Features
 
- `RoadType`, `Weather` — label-encoded to integers
- `LargeVehicles`, `Landmarks` — binarized
- `lanes_road_interact = NumberofLanes × RoadType_enc` — interaction term capturing combined road type and capacity effect
- `Temperature` NaNs imputed using median Temperature grouped by Weather type
### Final Feature Set (19 features)
 
```
hour_sin, hour_cos, min_sin, min_cos, day_sin, day_cos,
time_minutes, RoadType_enc, NumberofLanes, LargeVehicles_bin,
Landmarks_bin, Temperature, Weather_enc, geo_mean_demand,
geo_std_demand, geo_l4_mean, geo_l5_mean,
lanes_road_interact, peak_hour
```
 
---
 
## Step 5 — Optimization
 
Two ensemble models were trained to minimize the derived loss on `log(demand)`.
 
### Model 1 — Random Forest (RF)
 
- Ensemble of decision trees, each trained on a bootstrap sample with random feature subsets
- Minimizes MSE in log space implicitly through averaging
- Equivalent to a non-parametric MAP estimate with implicit regularization via tree depth constraints
- Key hyperparameters: `n_estimators`, `max_depth`, `min_samples_leaf`
### Model 2 — Gradient Boosting Machine (GBM)
 
- Sequentially adds trees to minimize the negative gradient of the loss (functional gradient descent)
- Explicitly minimizes MSE in log space (derived loss)
- L2 regularization controlled by `learning_rate` and `subsample` (stochastic gradient boosting)
### Ensemble Strategy
 
```python
final_log_pred = 0.55 × RF_prediction + 0.45 × GBM_prediction   # in log space
demand         = exp(final_log_pred)
demand         = clip(demand, 1e-7, 1.0)
```
 
Weights balance RF's stability (low variance) with GBM's precision (low bias), validated on held-out folds. Both models were trained using 5-fold cross-validation on 77,299 training samples.
 
---
 
## Step 6 — Validation
 
**Evaluation metric:** `score = max(0, 100 × R²(actual, predicted))`
 
R² is not the same as the training loss (MSE in log space). To ensure alignment:
 
- R² was validated directly on raw `demand` (after `exp` transform) during cross-validation — not just training loss
- Residual analysis was performed to check for systematic bias in specific geohash zones or time windows
- Prediction range verified: all values in `(0, 1]` as required
Cross-validation confirmed that minimizing MSE in log space reliably improves R² on the original demand scale.
 
---
 
## Results Summary
 
| Component | Choice | Rationale |
|---|---|---|
| Target transformation | `log(demand)` | Stabilizes variance under log-normal assumption |
| Loss function | MSE in log space + L1/L2 | Derived from MLE under log-normal; MAP with regularization |
| Model 1 | Random Forest | Low variance, stable baseline |
| Model 2 | Gradient Boosting | Low bias, precise functional gradient descent |
| Ensemble weights | RF 0.55 + GBM 0.45 | Validated on 5-fold CV |
| Final output | `clip(exp(ŷ), 1e-7, 1.0)` | Ensures valid demand range |
