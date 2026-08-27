# Research basis and architectural implications

This is a design bibliography, not a claim that any cited effect will make the final
VOO/KMLM allocator profitable. Each implication was translated into a small, testable
component and remains subordinate to final out-of-sample CAGR.

## Provider and index facts

KraneShares describes KMLM as benchmarked to the KFA MLM Index with 22 futures: 11
commodities, six currencies, and five global bond markets. It reports a 0.90% expense
ratio and a 2020-12-01 fund inception date. The project uses 2020-12-02 as its default
holdout boundary so that the first evaluated live-ETF return can begin from a completed
inception-day close. Sources: [official KMLM page](https://kraneshares.com/etf/kmlm/),
[official KMLM fact sheet](https://kraneshares.com/resources/factsheet/kmlm_factsheet.pdf),
and [official summary prospectus](https://kraneshares.com/resources/compliance/2024_08_01_kmlm_summary.prospectus.pdf).

Mount Lucas states that it introduced the MLM Index in 1988 as a passive measure of
systematic trend-following returns. This supports investigating long index economics but
does not make those returns live KMLM ETF returns. Source: [Mount Lucas MLM Index](https://www.mtlucas.com/mlm-index).

The provider describes the KFA index history as representations of the MLM Index for
1988–2004, MLM Index EV for 2005 through November 2020, and KFA MLM Index thereafter.
That is why the provenance template permits multiple explicit segments and rejects a
single silently relabeled ETF history. Source: [KraneShares managed-futures guide](https://kraneshares.com/managed-futures-etf-guide-what-it-is-how-it-works-and-why-kmlm/).

Vanguard reports that VOO tracks the S&P 500, began 2010-09-07, and has a 0.03% expense
ratio. Earlier economics therefore require a documented S&P 500 total-return series, not
fabricated VOO prices. Source: [official Vanguard VOO profile](https://investor.vanguard.com/investment-products/etfs/profile/voo).

Architectural implications:

- price-derived features dominate;
- the KMLM history is segmented by economic identity;
- gross index segments receive explicit fee drag;
- the final holdout requires actual KMLM ETF history;
- no provider performance chart is imported as backtest evidence.

## Return forecasting restraint

Welch and Goyal found broad instability and weak out-of-sample performance among many
equity-premium predictors. Their later update again finds that many published variables
lose significance or perform poorly out of sample. Sources: [2008 Review of Financial
Studies paper](https://academic.oup.com/rfs/article/21/4/1455/1565737) and [2024 update](https://academic.oup.com/rfs/article/37/11/3490/7749383).

Architectural implications:

- no large predictor zoo;
- only five optional external transforms;
- A/B/C/D ablations must prove incremental value;
- simple linear models remain first-class ensemble members;
- final success is a portfolio result, not forecast fit alone.

## Momentum and managed futures

Moskowitz, Ooi, and Pedersen document time-series momentum across liquid equity-index,
currency, commodity, and bond futures, with continuation primarily over one-to-twelve
month horizons and notable behavior in extreme markets. Source: [Time Series Momentum,
Journal of Financial Economics](https://pages.stern.nyu.edu/~lpederse/papers/TimeSeriesMomentum.pdf).

Architectural implications:

- 21/63/126/252-session trailing trend features are economically motivated;
- KMLM is modeled from its own behavior rather than assumed to be inverse VOO;
- relative momentum and ratio trend are directly decision-relevant;
- major-bear contribution is diagnosed but no crisis is hand-coded.

## Volatility timing

Moreira and Muir show that volatility-managed portfolios can have different risk/return
properties when exposure varies inversely with recent variance. This is not direct proof
of VOO/KMLM rotation alpha, and the project's objective is not Sharpe maximization.
Source: [Volatility Managed Portfolios, NBER Working Paper 22208](https://www.nber.org/papers/w22208).

Architectural implications:

- restrained volatility, downside-volatility, drawdown, and correlation features;
- forecast disagreement shrinks position changes;
- volatility diagnostics remain secondary unless they improve net CAGR.

## Nested evaluation and model-selection bias

Cawley and Talbot show that optimizing a noisy model-selection criterion can itself
overfit and bias performance estimates. Source: [JMLR model-selection paper](https://jmlr.org/papers/v11/cawley10a.html).

White's Reality Check addresses the chance that the best specification in a search appears
superior because the same history was reused. Hansen's SPA test is a related refinement.
Sources: [White 2000](https://onlinelibrary.wiley.com/doi/abs/10.1111/1468-0262.00152)
and [Hansen 2005](https://www.tandfonline.com/doi/abs/10.1198/073500105000000063).

Architectural implications:

- expanding outer folds and prequential inner folds;
- exact label-overlap purging;
- tiny hyperparameter grids;
- equal ensemble weights;
- a complete programmatic candidate ledger;
- circular-block Reality Check and CAGR-excess intervals;
- final single-use holdout after all selection.

The implemented Reality Check is a transparent practical version centered on candidate
daily excess returns. It is not represented as Hansen's SPA procedure or as a substitute
for the final holdout.

## Point-in-time data

FRED/ALFRED formalizes real-time periods and series vintage dates. These concepts are
necessary because a value visible now may not equal what a historical decision maker saw.
Sources: [FRED real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)
and [FRED vintage dates](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html).

Architectural implications:

- observation date, release date, vintage start, and vintage end are all required;
- the replay panel records what was visible at each close;
- current-vintage macro histories cannot silently enter Model D;
- external data are optional, never a core dependency.

## Probability calibration

Probabilities are evaluated by proper scores and reliability, not by directional accuracy
alone. The implementation uses chronological prequential predictions for a simple Platt
map and reports Brier score, log loss, expected calibration error, ROC AUC, and reliability
bins. Simple calibration was chosen to avoid fitting a flexible mapping to a small
financial sample.

## Evidence conclusion

The research supports a disciplined hypothesis test: price trends, relative behavior, and
a few point-in-time risk variables are plausible inputs. It does **not** establish that the
implemented allocator beats VOO. Only approved data run through the frozen protocol can
answer that question, and this distribution contains no such result.
