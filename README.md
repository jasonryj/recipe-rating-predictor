# Recipe Rating Predictor  
*EECS 398 Final Project – Jason Rong (jasonryj@umich.edu)*

---

## Introduction
The **Recipes & Ratings** dataset from Food.com contains ≈ 80 k recipes and 700 k user ratings.  
We parsed each recipe’s nutrition list into seven numeric columns and joined the mean user rating (`avg_rating`).  
Our goals:

1. **Explore** how nutrition and preparation factors relate to user ratings.  
2. **Predict** a recipe’s average rating from easy-to-obtain numeric features.

---

## Data Cleaning and Exploratory Data Analysis

**Cleaning rules**

* ⏱ Drop cook times > 600 min (1.24 % removed)  
* 🥗 Keep 0 < calories ≤ 3000 (0.8 % removed)  
* ✔️ Final: 82 072 recipes (97.9 % retained)

### Calories Distribution (Under 2000 kcal)
<iframe src="assets/calories_dist.html" width="800" height="500" frameborder="0"></iframe>

### Calories Distribution (Full)
<iframe src="assets/calories_cleaned.html" width="800" height="500" frameborder="0"></iframe>

### Average Rating Distribution
<iframe src="assets/rating_dist.html" width="800" height="500" frameborder="0"></iframe>

### Cook Time Distribution
<iframe src="assets/cooktime_dist.html" width="800" height="500" frameborder="0"></iframe>

---

### Rating by Calorie Bin
<iframe src="assets/rating_by_calorie_bin.html" width="800" height="500" frameborder="0"></iframe>

### Rating by Sugar Level
<iframe src="assets/rating_by_sugar.html" width="800" height="500" frameborder="0"></iframe>

### Rating by Sodium Level
<iframe src="assets/rating_by_sodium.html" width="800" height="500" frameborder="0"></iframe>

### Cook Time by Rating (Violin Plot)
<iframe src="assets/violin_cook_time.html" width="800" height="500" frameborder="0"></iframe>

### Rating by Number of Ingredients
<iframe src="assets/average_rating_by_ingredients.html" width="800" height="500" frameborder="0"></iframe>

---

## Framing a Prediction Problem
* **Target**: `avg_rating` (1 – 5 stars)  
* **Task**: Regression   *Metric*: RMSE  
* **Baseline features**: calories, minutes, n_ingredients  
* **Split**: 80 / 20, `random_state=888`

---

## Baseline Model

### Feature Correlation Heatmap (Interactive)
<iframe src="assets/corr_heatmap.html" width="800" height="600" frameborder="0"></iframe>

### Feature Correlation Heatmap (Static)
![Feature Correlation Heatmap](assets/basic.png)

### Average Rating by Group
![Average Rating](assets/rate_average.png)

*Linear Regression* on *(minutes, protein, carbohydrates)* → **Test RMSE = 1.086**  
Longer prep times and higher carbs slightly lower ratings; residual funnel suggests heteroscedasticity.

---

## Final Model

### TF-IDF Keyword Feature Importance (Interactive)
<iframe src="assets/word_feature.html" width="800" height="600" frameborder="0"></iframe>

### TF-IDF Keyword Feature Importance (Static)
![TF-IDF Word Importance](assets/final.png)

A Random Forest with engineered features (`log_minutes`, `carb_per_ing`) achieved **Test RMSE = 1.087**.  
Most influential numeric feature: **carb_per_ing**; top keywords in descriptions (e.g., *cooking*, *great*, *cheese*) correlate with higher ratings.

---

*Last updated: June 20 2025*
