# Recipe Rating Predictor  
*EECS 398 Final Project – Jason Rong (jasonryj@umich.edu)*  

---

## Introduction  
The **Recipes & Ratings** dataset from Food.com contains ≈ 80 k recipes and 700 k user ratings.  
We parsed each recipe’s *nutrition* list into seven numeric columns and joined the mean user rating (`avg_rating`).  

**Why this problem?**  
1. ★ **Interpretable target** – Average rating (1–5) is intuitive to chefs and users.  
2. 📊 **Rich data** – Numeric nutrition + text + time offers multiple modeling angles.  
3. 🍽 **Practical impact** – Identifying factors that drive high ratings can guide recipe creators and platforms.

Goals:  
* **Explore** how nutrition & prep factors relate to ratings.  
* **Predict** a recipe’s average rating from readily available numeric features.

---

## Data Cleaning and Exploratory Data Analysis  

| Step | What we did | Why? |
|------|-------------|------|
| **Remove cook time > 600 min** | `recipes = recipes[recipes['minutes'] <= 600]` | >10 h likely data-entry errors or niche slow-cook recipes that skew stats. |
| **Keep 0 < calories ≤ 3000** | `(0 < calories ≤ 3000)` | Negative/zero invalid; >3 000 kcal are rare outliers (<1 %). |
| **Split nutrition into 7 columns** | `for i, col in enumerate(nutrition_columns): ...` | Separate features for histograms, correlation, and models. |
| **Plot univariate histograms** | `px.histogram(...)` | Inspect skewness / extreme values; decide on log-scale if needed. |
| **Create calorie / sugar / sodium bins** | `pd.cut(...).groupby().mean()` | Capture non-linear rating trends versus nutrition levels. |
| **Violin plot log(minutes) vs rating** | `px.violin(...)` | Check variance & distribution shape across rating bands. |
| **Group by n_ingredients** | `groupby('n_ingredients').mean()` | Test whether “more ingredients ⇒ better rating” holds (diminishing returns after ~10). |

After cleaning we retained **82 072 recipes (97.9 %)**.

### Calories Distribution (< 2000 kcal)
<iframe src="assets/calories_dist.html" width="800" height="500" frameborder="0"></iframe>

### Calories Distribution (full)
<iframe src="assets/calories_cleaned.html" width="800" height="500" frameborder="0"></iframe>

### Average Rating Distribution
<iframe src="assets/rating_dist.html" width="800" height="500" frameborder="0"></iframe>

### Cook-Time Distribution
<iframe src="assets/cooktime_dist.html" width="800" height="500" frameborder="0"></iframe>

---

### Rating by Calorie Bin
<iframe src="assets/rating_by_calorie_bin.html" width="800" height="500" frameborder="0"></iframe>

### Rating by Sugar Level
<iframe src="assets/rating_by_sugar.html" width="800" height="500" frameborder="0"></iframe>

### Rating by Sodium Level
<iframe src="assets/rating_by_sodium.html" width="800" height="500" frameborder="0"></iframe>

### Cook-Time by Rating (violin)
<iframe src="assets/violin_cook_time.html" width="800" height="500" frameborder="0"></iframe>

### Rating by # Ingredients
<iframe src="assets/average_rating_by_ingredients.html" width="800" height="500" frameborder="0"></iframe>

---

## Framing a Prediction Problem  

| Item | Choice | Reason |
|------|--------|--------|
| **Target** | `avg_rating` (1-5, real) | Continuous label preserves nuance vs. classification. |
| **Task / Metric** | Regression · RMSE | RMSE penalises large star-errors; same unit as target. |
| **Baseline features** | calories, minutes, n_ingredients | Simple, universally recorded numeric fields. |
| **Train/Test split** | 80 / 20, `random_state=888` | Matches course spec; reproducible comparisons. |

---

## Baseline Model  

### Feature Correlation Heatmap (interactive)  
<iframe src="assets/corr_heatmap.html" width="800" height="600" frameborder="0"></iframe>

### Feature Correlation (static)  
![Feature Correlation Heatmap](assets/basic.png)

### Average Rating by Group (residual diagnostic)  
![Average Rating](assets/rate_average.png)

**Method:** StandardScaler ➜ LinearRegression on *(minutes, protein, carbohydrates)*  
*Test RMSE = 1.086*

*Why linear first?*  
* Fast to train, coefficients interpretable – establishes a performance floor.  
* VIF + OLS p-values revealed multicollinearity (calories ↔ carbs) and weak predictors, prompting feature reduction.

Residual funnel shows heteroscedasticity → need flexible model.

---

## Final Model  

### TF-IDF Keyword Importance (interactive)  
<iframe src="assets/word_feature.html" width="800" height="600" frameborder="0"></iframe>

### TF-IDF Keyword Importance (static)  
![TF-IDF Word Importance](assets/final.png)

**Pipeline:**  
1. Custom transformer adds `log_minutes` + `carb_per_ing` (carbs ÷ ingredients)  
2. `QuantileTransformer(output_distribution='normal')`  
3. `RandomForestRegressor` tuned via 3-fold GridSearch  
   *Best params*: n_estimators = 300, max_depth = 10, min_samples_leaf = 3  

| Metric | RMSE |
|--------|------|
| Train  | 1.053 |
| Test   | 1.087 |

**Why Random Forest?** Captures non-linear interactions; robust to outliers; needs minimal feature scaling.

Top numeric feature: **carb_per_ing** → “lower carb density” tends to score higher.  
TF-IDF shows words like *cooking*, *great*, *cheese* associate with higher ratings; generic terms (*easy*, *add*) trend lower.

---

## Conclusion & Next Steps  
* Numeric nutrition alone explains limited variance (RMSE ≈ 1).  
* Carb density and protein mildly boost ratings; extreme cook-times hurt.  
* Going further: integrate full ingredient text, description embeddings, or image cues; test gradient-boosted trees or large-language-model embeddings for better lift.

*Last updated: 20 June 2025*
