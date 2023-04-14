# Unreleased

- Moved all metrics in `metrics.cluster` except `metrics.Silhouette` to [river-extra](https://github.com/online-ml/river-extra).

## dist

- A `revert` method has been added to `stats.Gaussian`.
- A `revert` method has been added to `stats.Multinomial`.
- Added `dist.TimeRolling` to measure probability distributions over windows of time.

## imblearn

- Added `imblearn.ChebyshevUnderSampler` and `imblearn.ChebyshevOverSampler` for imbalanced regression.

## linear_model

- `linear_model.LinearRegression` and `linear_model.LogisticRegression` now correctly apply the `l2` regularization when their `learn_many` method is used.

## rules

- AMRules's `debug_one` explicitly indicates the prediction strategy used by each rule.
- Fix bug in `debug_one` (AMRules) where prediction explanations were incorrectly displayed when `ordered_rule_set=True`.

## stats

- A `revert` method has been added to `stats.Var`.
