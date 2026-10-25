---
name: demand-volatility-calculator
description: Calculates demand volatility (Standard Deviation / Mean) for each meal to identify items that need higher safety stock.
---
# Demand Volatility Calculator Skill
1. Group the cleaned training data by `meal_id`.
2. Compute the mean (`mean()`) and standard deviation (`std()`) of `num_orders` for each meal across all weeks.
3. Calculate the volatility ratio: `Volatility = Std / Mean`.
4. Rank meals from highest to lowest volatility and flag top volatile meals that require buffer stock.
