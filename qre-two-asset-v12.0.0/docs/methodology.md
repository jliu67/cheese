# Methodology

## Research question and success rule

The only primary comparison is the net dynamic portfolio versus net buy-and-hold VOO on
the same decision-aligned dates. Success requires both:

\[
\operatorname{CAGR}_{QRE} > \operatorname{CAGR}_{VOO}
\]

\[
W^{QRE}_{T} > W^{VOO}_{T}
\]

The engine also requires pre-holdout rolling persistence and a multiple-testing check for
deployment. Drawdown, Sharpe, Sortino, Calmar, and other risk measures are secondary.

## Feature definitions

All return features are log returns from continuous net total-return levels. Every rolling
calculation is trailing and includes only data through the signal close.

### Variant A — VOO only (12)

| Feature | Definition |
|---|---|
| `voo__log_return_{21,63,126,252}` | `log(level_t / level_{t-h})` |
| `voo__vol_21`, `voo__vol_63` | Annualized population standard deviation of daily log returns |
| `voo__downside_vol_63` | Annualized standard deviation after nonnegative returns are set to zero |
| `voo__drawdown_252` | Level divided by trailing 252-session maximum minus one |
| `voo__ma_gap_63`, `voo__ma_gap_200` | Level divided by trailing arithmetic mean minus one |
| `voo__downside_frequency_21` | Fraction of negative simple-return sessions |
| `voo__momentum_accel_21_63` | 21-session log return minus one-third of 63-session log return |

### Variant B — KMLM additions (11; 23 cumulative)

KMLM adds the same four log-return horizons, 21/63-session volatility,
63-session downside volatility, 252-session drawdown, 63/200-session moving-average
gaps, and 21-session downside frequency. Momentum acceleration is omitted to keep the
increment restrained.

### Variant C — relative additions (12; 35 cumulative)

Let `L_t = log(VOO_t) - log(KMLM_t)`.

| Feature | Definition |
|---|---|
| `relative__log_return_{21,63,126,252}` | `L_t - L_{t-h}` |
| `relative__ratio_slope_63`, `_126` | Change in `L` divided by 62 or 125 sessions |
| `relative__vol_63` | Annualized volatility of daily changes in `L` |
| `relative__correlation_63` | Trailing correlation of VOO and KMLM daily log returns |
| `relative__correlation_change_21` | 21-session change in trailing correlation |
| `relative__drawdown_252` | Drawdown of the VOO/KMLM level ratio |
| `relative__risk_adjusted_momentum_63`, `_126` | Relative log return divided by relative volatility |

### Variant D — external additions (5; 40 cumulative)

| Feature | Definition |
|---|---|
| `external__vix_z_252` | Trailing 252-session z-score, clipped to ±8 |
| `external__vix_change_21` | 21-session change |
| `external__hy_oas_z_252` | Trailing 252-session z-score, clipped to ±8 |
| `external__hy_oas_change_21` | 21-session change |
| `external__nfci_change_20` | 20-session change |

Minimum rolling observations are encoded in `features.py`. Missing warm-up rows are not
backfilled. Model rows require at least 65% feature coverage; imputation is fitted inside
each training boundary.

## Targets

For a signal after close `t`, execution lag `e=1`, and horizon `h`:

\[
y^{rel}_{t,h} =
\left(\log VOO_{t+e+h}-\log VOO_{t+e}\right)
-
\left(\log KMLM_{t+e+h}-\log KMLM_{t+e}\right)
\]

The classification target is `1[y_rel > 0]`. Separate VOO and KMLM forward log returns
are retained in the target object for diagnostics. Horizons are frozen at 21, 63, and 126
sessions. The label end date is stored for exact fold purging.

## Nested chronological validation

The pre-holdout sample uses expanding annual outer folds after an eight-year minimum
training period. Each outer training set is sampled at the last available session of each
week for model fitting, reducing redundant overlapping observations. Three inner
prequential folds select tiny hyperparameter grids:

| Model | Candidate parameter |
|---|---|
| Ridge | alpha 1 or 10 |
| Logistic | C 0.1 or 1 |
| HGB regression | maximum leaves 7 or 15 |
| HGB classification | maximum leaves 7 or 15 |

The tree models use learning rate 0.04, 120 iterations, minimum 30 observations per leaf,
and L2 regularization. They are intentionally shallow. The hyperparameter score is MSE
for regression and Brier loss for classification. No random train/test split is available.

All training labels must end strictly before the test start. Scaling, imputation,
hyperparameter selection, model fitting, and calibration remain inside training history.
Outer predictions are daily so the allocation cadence can be compared independently of
the weekly model-fitting sample.

## Probability calibration and uncertainty

The two classifier probabilities are averaged, transformed to log odds, and calibrated by
a one-variable logistic map fitted on prequential predictions. Calibration evaluation uses
a later chronological tail than the data used to fit the evaluation calibrator. The final
calibrator is then refit on all mature prequential predictions for deployment.

Artifacts report Brier score, log loss, expected calibration error, ROC AUC, and a ten-bin
reliability curve per horizon. Regression residual 10th/90th quantiles produce an empirical
80% prediction interval. Regressor disagreement, classifier disagreement, and residual
scale mechanically shrink the allocation score. These intervals are empirical research
diagnostics, not guaranteed coverage bounds.

## Horizon ensemble

Default horizon weights are 0.20, 0.35, and 0.45 for 21, 63, and 126 sessions. Relative
return forecasts and interval bounds are rescaled to a 63-session equivalent. Classifier
probabilities and standardized regression scores are averaged with the same fixed weights.
No weights are optimized continuously.

The final score combines a bounded standardized-return term and a probability-direction
term equally, then applies the model-agreement shrinkage. The score remains in `[-1,1]`.
The stress report also evaluates equal and single-horizon weight sets without refitting.

## Allocation and turnover control

The continuous mapping is:

\[
w^{VOO}_t = \operatorname{clip}(w_0 + m s_t, 0, 1)
\]

where default neutral VOO weight `w0=0.70`, maximum tilt `m=0.70`, and score `s` is the
ensemble output. Bucketed allocation maps this target to the nearest frozen bucket.
Probability allocation maps classifier probability 0.35 to 0% VOO and 0.65 to 100% VOO,
with clipping.

The research grid compares daily and weekly execution opportunities and 5%/10% minimum
allocation changes. A target is accepted only when its absolute change meets the threshold.
Within 25 annual basis points of the highest pre-holdout CAGR, lower turnover wins.

All-in transaction cost is applied to one-way portfolio turnover:

\[
\text{turnover}_t = \frac{1}{2}\sum_i |w_{i,t}-w_{i,t-1}|
\]

\[
r^{net}_t = \sum_i w_{i,t}r_{i,t} - \text{turnover}_t\frac{c}{10{,}000}
\]

The default `c=5` bps should be replaced or stressed with a defensible combined estimate
of spread, commissions, market impact, and slippage. The engine assumes the portfolio
begins at 100% VOO.

## Candidate selection and data snooping

Outer predictions are research-development OOS, but selecting a feature variant and
allocator on those predictions still creates selection bias. The system therefore:

- logs every feature variant and allocator candidate;
- retains all candidate return series;
- reports a circular-block implementation of White's Reality Check versus VOO;
- reports a circular-block confidence interval for CAGR excess;
- treats the recent holdout as the only untouched final evaluation.

The Reality Check implementation is intentionally labeled as a practical approximation;
it is not presented as an SPA test. Its block length defaults to 21 sessions.

## Primary and secondary reporting

Primary metrics are CAGR, terminal wealth, annualized excess over VOO, cumulative excess
wealth, and rolling 3/5/10-year excess-return distributions and win rates.

Secondary metrics are maximum drawdown, annualized volatility, Sharpe, Sortino, Calmar,
downside deviation, ulcer index, worst 21/63/252-session return, maximum underwater
duration, maximum trough-to-recovery duration, turnover, and cost.

Static monthly-rebalanced benchmarks are VOO, KMLM, 50/50, 60/40, 70/30, and 80/20.
The holdout report says `FAILED` whenever the frozen alpha gate fails, regardless of
secondary metrics.

## Diagnostics and stress tests

Every run saves:

- A/B/C/D grouped ablation results;
- fold-wise feature coverage, median, and IQR;
- feature pairs with absolute pre-holdout correlation at least 0.95;
- holdout permutation importance (reporting only, never selection);
- live median-reset contributions (mechanical local sensitivity, not causal SHAP values);
- false defensive-signal frequency, duration, and approximate opportunity cost;
- excess log-wealth contribution across price-defined bull/calm, correction, major-bear,
  ordinary, and high-volatility non-bear periods;
- fixed historical episode tables for dot-com, GFC, 2011, 2015–16, Q4 2018, COVID,
  2020–21, 2022, and later markets, used only as hindsight diagnostics;
- bear-timing tables with first VOO reduction, VOO loss before reduction, KMLM return
  after reduction, first VOO re-entry, recovery missed, and net wealth versus VOO;
- cost/threshold grids, alternate cadence, ±10% forecast strength, an extra execution
  session, and shifted sample endpoints;
- base, equal, and single-horizon ensembles.

Outer expanding windows and inner candidate tables expose training-window and
hyperparameter stability. The chosen estimators use no stochastic subsampling or early
stopping; an automated alternate-seed test asserts identical predictions. Full
alternative-lookback research must be run as a separately
logged development experiment and must never be initiated after final-holdout inspection.
