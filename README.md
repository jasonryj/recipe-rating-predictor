# Recipe Rating Predictor  
*EECS 398 Final Project – Jason Rong (jasonryj@umich.edu)*  

---

## Introduction  
The **Recipes & Ratings** dataset from Food.com contains roughly **80 000 recipes** and **700 000 user ratings**.  
We parsed each recipe’s `nutrition` list into seven numeric columns and joined the mean user rating (`avg_rating`).  

### Why study this?  
1. **Interpretable target** – Star rating (1-5) is intuitive to both chefs and users.  
2. **Rich feature space** – Numeric nutrition, prep time, ingredients, and text allow multi-angle analysis.  
3. **Practical value** – Knowing drivers of high ratings can guide recipe ranking or health-oriented suggestions.  

---

### Dataset Introduction  
Each recipe includes metadata (`name`, `minutes`, `tags`), a list of ingredients and steps, and a **`nutrition`** array that we split into: calories, fat, sugar, sodium, protein, saturated fat, and carbohydrates.

To understand nutritional composition we first inspected **calorie distribution**. Most recipes are under 500 kcal, but a long right tail reaches ~2 000 kcal.  
This breakdown sets the stage for exploring how nutrition relates to user ratings and prep time.

---

## Data Cleaning and Exploratory Data Analysis  

| Step | What we did | Why |
|------|-------------|-----|
| ⏱ **Remove cook time > 600 min** | `recipes = recipes[recipes['minutes'] <= 600]` | >10 h likely data entry errors; extreme outliers distort averages. |
| 🥗 **Keep 0 < calories ≤ 3000** | `(0 < calories <= 3000)` | Negative / 0 invalid; >3 000 kcal are <1 % extreme dishes. |
| 🔎 **Split nutrition into 7 columns** | loop over `nutrition` list | Needed for histograms, correlations, models. |
| 📊 **Univariate & bivariate plots** | histograms, bins, violin | Visual check of skew, outliers, non-linear effects. |
| 🧑‍🍳 **Group by `n_ingredients`** | mean rating vs ingredients | Test if “more ingredients ⇒ better rating.” |

> **After cleaning:** **82 072 recipes (97.9 % retained).**

*Why plot?* Establishes baseline calorie distribution and motivates trimming extreme calories in cleaning.
#### Calories Distribution
<iframe src="assets/calories_cleaned.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Most recipes cluster below **500 kcal**; the long right-tail highlights a handful of high-energy dishes.  

*Why plot?* Establishes baseline calorie distribution and motivates trimming extreme calories in cleaning.


#### Average Rating Distribution
<iframe src="assets/rating_dist.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Bimodal peaks at ★4 and ★5 reveal user rating bias toward higher scores.  

*Why plot?* Helps choose RMSE (skewed, bounded target) and alerts us that class imbalance is minor.
#### Cook-Time Distribution
<iframe src="assets/cooktime_dist.html" width="800" height="500" frameborder="0"></iframe>

*Interpretation.* Heavy right-skew; median ≈ 40 min, few > 600 min.  

*Why plot?* Supports log-transform (`log_minutes`) and cook-time ≤ 600 min cleaning rule.

#### Rating by Calorie Bin
<iframe src="assets/rating_by_calorie_bin.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Average rating dips slightly as calories exceed 1000 kcal.  

*Why plot?* Shows non-linear nutrition effect; motivates treating calories via bins or tree models.
#### Rating by Sugar Level
<iframe src="assets/rating_by_sugar.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Little variation—users don’t penalise sugar strongly within common ranges.  

*Why plot?* Indicates sugar may be dropped from baseline features.
#### Rating by Sodium Level
<iframe src="assets/rating_by_sodium.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Slight uptick at very high sodium but overall flat.  

*Why plot?* Confirms sodium’s limited predictive value in numeric-only model.
#### Cook-Time by Rating (violin)
<iframe src="assets/violin_cook_time.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Higher-rated recipes show tighter, generally shorter log-cook-time.  

*Why plot?* Suggests minutes is predictive and log-scale reduces heteroscedasticity.
#### Rating by Number of Ingredients
<iframe src="assets/average_rating_by_ingredients.html" width="800" height="500" frameborder="0"></iframe>
*Interpretation.* Ratings plateau after ~10 ingredients; very complex recipes don’t gain stars.  

*Why plot?* Guides feature engineering—ingredient count has diminishing returns.
---

## Framing a Prediction Problem  
| Item | Choice | Reason |
|------|--------|--------|
| **Target** | `avg_rating` (1.0 – 5.0) | Continuous star score keeps nuance vs. binning. |
| **Task / Metric** | Regression • **RMSE** | RMSE penalises large star-errors; same unit as target. |
| **Baseline features** | calories, minutes, n_ingredients | Simple numeric fields available for every recipe. |
| **Split** | 80 % train / 20 % test, `random_state = 888` | Reproducible & aligns with project spec. |

---

## Baseline Model  

### Feature Correlation Heatmap (interactive)
<iframe src="assets/corr_heatmap.html" width="800" height="600" frameborder="0"></iframe>

### Feature Correlation Heatmap (static)
![Feature Correlation Heatmap](assets/basic.png)

### Regression Feature Selection Summary  
Based on **OLS** and **VIF** we keep **minutes, protein, carbohydrates**; drop calories (high VIF) and other low-value predictors.  
*Interpretation.* High collinearity between **calories & carbs** (r≈0.73) and other pairs.  

*Why plot?* Drives VIF analysis and decision to drop multicollinear calories in baseline model.
### Average Rating by Group (diagnostic)
![Average Rating](assets/rate_average.png)

*StandardScaler → LinearRegression* on the three features  
**Train RMSE = 1.083 • Test RMSE = 1.086**

Residual funnel (heteroscedasticity) shows the linear model under-predicts many 5-star recipes → motivates non-linear models.

---

## Final Model  

*Pipeline:*  
1. Add engineered features: `log_minutes`, **carb_per_ing** (carbs ÷ ingredients)  
2. `QuantileTransformer(output_distribution='normal')`  
3. **RandomForestRegressor** tuned (300 trees • max_depth 10 • min_samples_leaf 3)

| Metric | RMSE |
|--------|------|
| Train  | **1.053** |
| Test   | **1.087** |

Top numeric importance: **carb_per_ing > protein > carbohydrates**.

### TF-IDF Keyword Importance (interactive)
<iframe src="assets/word_feature.html" width="800" height="600" frameborder="0"></iframe>

### TF-IDF Keyword Importance (static)
![TF-IDF Word Importance](assets/final.png)

*Interpretation.* Words like *cooking, great, cheese* align with higher ratings; generic verbs lower.  

*Why plot?* Demonstrates added explanatory power of text features beyond numeric nutrition.

---

## Conclusion & Next Steps  
* Numeric nutrition explains limited variance (RMSE ≈ 1).  
* “Lower carb density + higher protein” and moderate prep time correlate with better ratings.  
* Future work: integrate full ingredient/description text, image quality, and test gradient-boost models or transformer embeddings for improved accuracy.

*Last updated: 20 June 2025*

