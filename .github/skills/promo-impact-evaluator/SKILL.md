---
name: promo-impact-evaluator
description: Evaluates the impact of promotions and homepage featuring on food orders using statistical testing (Mann-Whitney U test) and correlation.
---
# Promo Impact Evaluator Skill
1. Filter the dataset to compare the same meal in the same center with and without promotion (`emailer_for_promotion` or `homepage_featured`).
2. Since order counts are skewed, apply the Mann-Whitney U test (`scipy.stats.mannwhitneyu`) to check if the difference is statistically significant.
3. Calculate the percentage difference in median orders between promoted and non-promoted periods.
4. Output a summary table and verify p-values against the 0.05 threshold.