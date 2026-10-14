# Evaluation report

Selected on validation log loss: **logistic_regression**. Test season: **2025/26** (380 matches).

| Model | Accuracy | Log loss ↓ | Brier ↓ |
|---|---:|---:|---:|
| prior_baseline | 42.6% | 1.0851 | 0.6565 |
| logistic_regression | 50.3% | 1.0296 | 0.6183 |
| gradient_boosting | 49.2% | 1.0550 | 0.6324 |

## Protocol

- Warm-up: 2015/16. Training: 2016/17–2023/24 (3040 matches).
- Model selection: 2024/25 (380 matches), with fixed candidate settings.
- Refit on training + validation, then evaluate on 2025/26. Test scores do not select the winner.
- Parameters stay frozen throughout the test season; form and Elo update after each completed match date. This is sequential pre-match evaluation, not a season-ahead forecast.
- Same-day fixtures share the state from before that date. No current-match goals, shots, odds or outcome enter features.
- Elo starts at 1500, K=20, home adjustment=60; ratings regress 25% toward 1500 each season. Form uses the last five games within that season; missing form uses fixed priors.
- Brier is the mean sum of squared errors across three classes (range 0–2). Log loss is the main probability metric. Lower is better for both.
- Calibration curves are descriptive and noisy for 380 matches. No probability calibration was fitted on the test data.

![Evaluation](evaluation.png)

## Limits

One held-out season is limited evidence. No injuries, lineups, transfers or bookmaker information are used. Promoted teams without Premier League history start at the default Elo. These are estimates, not guaranteed outcomes. Draw recall can be poor even when probability metrics improve. No profit or betting-performance claim is made.

The separate demo model is refitted on all eligible seasons only AFTER evaluation. Its in-sample outputs must not be presented as test performance. Demo state ends on 2026-05-24; it is a historical snapshot, not a live fixture forecast.

Source: [Football-Data](https://www.football-data.co.uk/englandm.php). Exact downloaded file hashes are in `data/manifest.json`.
