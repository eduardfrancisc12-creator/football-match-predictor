# Model card

**Task:** three-class pre-match outcome probabilities for English Premier League football.

**Intended use:** learning, portfolio demonstrations and exploration of historical team form. Not intended for live operational forecasting, betting decisions or claims of guaranteed results.

**Data:** Football-Data Premier League CSV results for 2015/16–2025/26. Goals are used to construct labels and future historical features, never as current-match inputs. No personal data is required.

**Selection:** class-prior baseline, logistic regression and histogram gradient boosting. Fixed configurations. Lowest log loss on 2024/25 determines the selected model. All candidates are evaluated on the same later 2025/26 dates; those results do not change selection.

**Evaluation:** chronological, sequential pre-match simulation. Earlier completed fixtures update state; learned model parameters stay fixed. Probability quality is measured using log loss and multiclass Brier score; accuracy, confusion matrix and reliability curves are supplementary. Actual scores and model identity are in `reports/metrics.json`.

**Artifact separation:** `evaluation_model.joblib` is fitted only through 2024/25. `demo_model.joblib` is a separate refit through 2025/26 and is not the evaluated holdout model. `state.json` contains final historical state, cutoff and team options.

**Known limitations:** one holdout season; limited team features; no injuries, transfers or lineups; default ratings for teams absent from history; limited draw recall; no formal uncertainty intervals on reported metrics. Outcomes are noisy and estimates may be miscalibrated or drift over time.

**Update policy:** adding new data requires rerunning validation and reporting the new cutoff. Changes designed after seeing 2025/26 performance must not claim that season remains an untouched final test. Use a new holdout or clearly label retrospective backtests.
