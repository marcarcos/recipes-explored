# recipes-explored
## Introduction

**Dataset:** A collection of ~80 K user‐submitted recipes, each with ingredients, tags, cooking time (`minutes`), nutrition facts, and user ratings/reviews.

**Project Question:**  
Do unreviewed recipes differ systematically in their characteristics (e.g., cooking time, nutritional profile) from recipes that have been reviewed? And can we build a recommender that balances textual and nutritional similarity?

- **Rows:** 83 861 recipes  
- **Relevant columns:**  
  - `minutes`: cooking time in minutes  
  - `review_count`: number of user reviews (missing if zero)  
  - `avg_rating`: average user rating (1–5 scale)  
  - Text fields: `ingredients`, `tags`, `description`  
  - `nutrition`: nutrition formatted by 
  - Engineered flags: `no_reviews`, `log10_min`, `is_alcohol`, etc.

**Why it matters:**  
Understanding biases in reviews helps surface under‐reviewed yet worthwhile recipes, and a balanced recommender can suggest dishes that are both textually and nutritionally aligned with user preferences.

## Data Cleaning and Exploratory Data Analysis

1. **Cleaning steps**  
   - Parsed `ingredients`, `tags`, `description` into a cleaned “corpus” for TF-IDF.  
   - Converted numeric nutrition DV columns into per-calorie ratios (`fat_dv_to_cal`, etc.).  
   - Flagged missing reviews (`no_reviews`) and log-transformed cooking time (`log_minutes`).  
   - Binned cooking time into log-spaced categories (`time_log_bin`).

2. **Univariate analysis**  
   <iframe
     src="assets/univariate_minutes_hist.html"
     width="800" height="400"
     frameborder="0">
   </iframe>  
   _Figure:_ Distribution of cooking times. Most recipes cook in 2–119 min.

3. **Bivariate analysis**  
   <iframe
     src="assets/bivariate_rating_vs_minutes.html"
     width="800" height="400"
     frameborder="0">
   </iframe>  
   _Figure:_ Scatter of `avg_rating` vs. `minutes`. Slight downward trend for very long recipes.

4. **Interesting aggregate**  
   ```markdown
   | time_bin | % no reviews | n_recipes |
   |:---------|-------------:|----------:|
   | 0–2      |        0.00% |      1 137 |
   | 2–119    |        6.54% |     73 383 |
   | 119–5 854|        8.72% |      9 176 |

We found that `review_count` is missing when a recipe has never been reviewed. Grouping by log-spaced cooking-time bins:

| Time to Make (min) | % No Reviews | # Recipes |
|:-------------------|-------------:|----------:|
| 0–2                |       0.00%  |     1 137 |
| 2–119              |       6.54%  |    73 383 |
| 119–5 854          |       8.72%  |     9 176 |
| 5 854–288 000      |       0.00%  |        84 |

<iframe
  src="assets/missing_by_logtime.html"
  width="800" height="400"
  frameborder="0">
</iframe>

**Interpretation:**  
Missingness in `review_count` clearly depends on cooking time (not MCAR). It is at least MAR given `time_log_bin`, and possibly NMAR if very long recipes simply aren’t tried or reviewed.

## Hypothesis Testing

We tested whether recipes with zero reviews cook longer than those with ≥1 review.

- **H₀:** No difference in mean cooking time between zero-review and reviewed recipes.  
- **H₁:** Zero-review recipes take longer on average.  
- **Observed Δ mean:** 40.76 min (zero-review minus reviewed)  
- **Permutation p-value:** 0.0790  

<iframe
  src="assets/perm_null_distribution.html"
  width="800" height="500"
  frameborder="0">
</iframe>

**Conclusion:**  
p = 0.0790 > 0.05, so we fail to reject H₀. There’s only weak evidence that unreviewed recipes cook longer.

---

**Robust test (medians):**

- **Observed Δ median:** 5.00 min  
- **Permutation p-value:** 0.3574  

<iframe
  src="assets/perm_median_null.html"
  width="800" height="500"
  frameborder="0">
</iframe>

**Conclusion:**  
Even by medians, p = 0.3574 > 0.05, so no evidence of longer cook times for zero-review recipes.

## Framing a Prediction Problem

**Task:** Recommend similar recipes given a query recipe.  
- **Type:** Information‐retrieval “similarity” problem (nearest‐neighbor).  
- **Response:** Top-k similar recipes by cosine similarity.  
- **Evaluation metric:** Average nearest-neighbor cosine similarity on a held-out set (higher is better).  
- **Features available at prediction time:**  
  - TF-IDF on `corpus` (ingredients, tags, description)  
  - Scaled numeric features: `log_minutes`, nutrition ratios, etc.  

## Baseline Model

We built a baseline KNN using:

- **Features:**  
  - TF-IDF (top 2 000 terms)  
  - Numeric: `log_minutes`, `fat_dv_to_cal`, `protein_dv_to_cal`  
- **Pipeline:** Single `NearestNeighbors` fit on hstack([TF-IDF, numeric])  
- **Held-out evaluation:**  
  - Average similarity ≈ 0.7500  

The baseline confirms that text+basic nutrition yields moderate neighbor quality.

## Final Model

We engineered additional nutrition features (`sugar`, `cholesterol_mg_dv`, `some_ferm`) and tuned weights:

- **Weight α** on new numeric block: **8.0**  
- **Text weight β:** 1.0 (baseline)  
- **Held-out avg similarity:** 0.8610 (↑ 15% over baseline)

<iframe
  src="assets/alpha_tuning_plot.html"
  width="800" height="400"
  frameborder="0">
</iframe>

**Example recommendations:**  

```python
recommend(42, k=5) → [ … list of recipe names … ]
