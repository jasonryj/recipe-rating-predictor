# Recipe Rating Predictor  
*EECS 398 Final Project – Jason Rong (jasonryj@umich.edu)*  

---

## Introduction  
The **Recipes & Ratings** dataset from Food.com contains ~ 80 k recipes and 700 k user ratings.  
We split each recipe’s `nutrition` array into seven numeric columns and merged the mean user rating (`avg_rating`).  

**Why study this?**  

1. **Interpretable target** – a 1-to-5 star score is intuitive for both cooks and users.  
2. **Rich feature space** – numeric nutrition, prep time, ingredient count, and text enable multi-angle analysis.  
3. **Practical value** – discovering drivers of high ratings can guide recipe ranking & health suggestions.  

### Dataset Introduction  
Each recipe provides metadata (`name`, `minutes`, `tags`), ingredient list, steps, and a `nutrition` vector (calories, fat, sugar, sodium, protein, saturated fat, carbohydrates).

Most recipes fall below 500 kcal, but a long right tail approaches 2 000 kcal – a key motivation for outlier trimming.

---

## Data Cleaning & Exploratory Data Analysis  

| Step | Action | Rationale |
|------|--------|-----------|
| ⏱ Remove *minutes* > 600 | Filter extreme cook times | > 10 h entries are likely data errors and distort averages. |
| 🥗 Keep 0 < calories ≤ 3000 | Drop invalid & ultra-high calories | Ensures realistic energy values (affects < 1 % rows). |
| 🔎 Split nutrition vector | Seven separate columns | Required for histograms, correlation, models. |
| 📊 Plot univariate / bivariate distributions | Histograms, bins, violin plots | Inspect skew, outliers & non-linear patterns. |
| 🧑‍🍳 Group by `n_ingredients` | Mean rating vs. ingredient count | Test whether “more ingredients ⇒ higher rating.” |

> **After cleaning:** **82 072 recipes (97.9 % retained).**

### Calories Distribution
<iframe src="assets/calories_cleaned.html" width="800" height="500" frameborder="0"></iframe>

> **Interpretation.** Most dishes cluster < 500 kcal; extreme high-calorie recipes are rare.  
> **Purpose.** Confirms the 0–3000 kcal filter and establishes baseline energy profile.

### Average Rating Distribution
<iframe src="assets/rating_dist.html" width="800" height="500" frameborder="0"></iframe>

> **Interpretation.** Bimodal peaks at ★4 and ★5 reveal user bias toward high ratings.  
> **Purpose.** Guides choice of RMSE (bounded, skewed target).

### Cook-Time Distribution
<iframe src="assets/cooktime_dist.html" width="800" height="500" frameborder="0"></iframe>

> **Interpretation.** Heavy right-skew; median ≈ 40 min, very few > 600 min.  
> **Purpose.** Justifies log-transform (`log_minutes`) and cook-time cap.

### Rating by Calorie Bin
<iframe src="assets/rating_by_calorie_bin.html" width="800" height="500" frameborder="0"></iframe>

> **Interpretation.** Average rating drifts downward after ~ 1000 kcal.  
> **Purpose.** Shows a mild nutrition effect and motivates tree-based models.

### Rating by Sugar Level
<iframe src="assets/rating_by_sugar.html" width="800" height="500" frameborder="0"></iframe>

> Users hardly penalize sugar within common ranges → sugar dropped from baseline.

### Rating by Sodium Level
<iframe src="assets/rating_by_sodium.html" width="800" height="500" frameborder="0"></iframe>

> Sodium shows minimal signal; confirms its low predictive value.

### Cook-Time by Rating (violin)
<iframe src="assets/violin_cook_time.html" width="800" height="500" frameborder="0"></iframe>

> High-rated recipes cluster around shorter *log-minutes* with tighter variance → minutes is predictive.

### Rating by Number of Ingredients
<iframe src="assets/average_rating_by_ingredients.html" width="800" height="500" frameborder="0"></iframe>

> Ratings plateau after ~ 10 ingredients → “more isn’t always better.”

---

## Framing a Prediction Problem  

| Item | Choice | Reason |
|------|--------|--------|
| **Target** | `avg_rating` (1–5) | Continuous stars preserve nuance. |
| **Metric** | **RMSE** | Same unit as target; penalizes large errors. |
| **Baseline features** | calories · minutes · n_ingredients | Simple, universally present. |
| **Split** | 80 % train / 20 % test (`random_state=888`) | Reproducible and per-spec. |

---

## Baseline Model  

### Correlation Heatmap (interactive)
<iframe src="assets/corr_heatmap.html" width="800" height="600" frameborder="0"></iframe>

![Static Heatmap](assets/basic.png)

> High collinearity between **calories & carbs** (r ≈ 0.73) → drop calories via VIF screening.

### Regression Feature Selection Summary  
Keep `minutes`, `protein`, `carbohydrates`; drop calories & low-impact nutrients.  

![Average Rating Diagnostic](assets/rate_average.png)

*StandardScaler → LinearRegression*  
**Train RMSE 1.083 • Test RMSE 1.086**  

> Residual funnel indicates heteroscedasticity → need a non-linear model.

---

## Final Model  

**Pipeline**  

1. Engineered features: `log_minutes`, **carb_per_ing** (carbs ÷ ingredients)  
2. `QuantileTransformer` → normal  
3. **RandomForestRegressor** (`300` trees · `max_depth=10` · `min_samples_leaf=3`)

| Metric | RMSE |
|--------|------|
| Train | **1.053** |
| Test  | **1.087** |

> Top importance: **carb_per_ing > protein > carbohydrates** – lower carb density & higher protein trend better.

### TF-IDF Keyword Importance (interactive)
<iframe src="assets/word_feature.html" width="800" height="600" frameborder="0"></iframe>

![Static TF-IDF](assets/final.png)

> Words like *cooking*, *great*, *cheese* align with high ratings; generic verbs (*easy*, *use*) align lower → text adds explanatory power.

---

## Conclusion & Next Steps  

* Numeric nutrition alone explains limited variance (RMSE ≈ 1).  
* “Lower carb density + higher protein” and moderate prep times correlate with better ratings.  
* Next: integrate ingredient/description text embeddings, photos, and test gradient-boosted models or transformers for further gains.

*Last updated: 20 Jun 2025*

