"""Select on 2024/25; evaluate once on 2025/26; then refit a separate demo model."""
import json
from pathlib import Path
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from football.data import ROOT, load_matches
from football.features import FEATURES, build_features

LABELS = ['A', 'D', 'H']


def candidates():
    return {
        'prior_baseline': DummyClassifier(strategy='prior'),
        'logistic_regression': make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=42)),
        'gradient_boosting': HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, min_samples_leaf=40, learning_rate=.05, l2_regularization=5., early_stopping=False, random_state=42),
    }


def scores(model, frame):
    p = model.predict_proba(frame[FEATURES])
    if list(model.classes_) != LABELS:
        raise ValueError('Unexpected probability column order')
    y = frame.FTR
    onehot = np.array([[float(label == value) for label in LABELS] for value in y])
    return {'accuracy': float(accuracy_score(y, np.array(LABELS)[p.argmax(axis=1)])), 'log_loss': float(log_loss(y, p, labels=LABELS)), 'brier_multiclass': float(np.mean(np.sum((p - onehot) ** 2, axis=1)))}, p


def main():
    matches = load_matches()
    frame, state = build_features(matches)
    # First season warms up team ratings; it is never a training observation.
    eligible = frame[frame.Season != '1516'].copy()
    train = eligible[eligible.Season < '2425']
    val = eligible[eligible.Season == '2425']
    test = eligible[eligible.Season == '2526']
    assert train.Date.max() < val.Date.min() <= val.Date.max() < test.Date.min()
    validation = {}
    for name, model in candidates().items():
        model.fit(train[FEATURES], train.FTR)
        validation[name], _ = scores(model, val)
    # Baseline is eligible to win: no forcing a more complex model to look good.
    selected = min(validation, key=lambda name: validation[name]['log_loss'])
    fitting = pd.concat([train, val])
    results, predictions = {}, {}
    for name, model in candidates().items():
        model.fit(fitting[FEATURES], fitting.FTR)
        results[name], predictions[name] = scores(model, test)
        if name == selected:
            evaluation_model = model
    report_dir = ROOT / 'reports'
    artifact_dir = ROOT / 'artifacts'
    report_dir.mkdir(exist_ok=True)
    artifact_dir.mkdir(exist_ok=True)
    metadata = {'selected_model': selected, 'selection_metric': 'validation log loss', 'validation_season': '2024/25', 'test_season': '2025/26', 'training_matches': len(train), 'validation_matches': len(val), 'test_matches': len(test), 'total_matches': len(matches), 'data_cutoff': str(matches.Date.max().date()), 'validation': validation, 'test': results, 'features': FEATURES, 'labels': LABELS, 'python': platform.python_version(), 'sklearn': sklearn.__version__}
    (report_dir / 'metrics.json').write_text(json.dumps(metadata, indent=2))
    output = test[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].reset_index(drop=True)
    for name, prob in predictions.items():
        for j, label in enumerate(LABELS):
            output[f'{name}_{label}'] = prob[:, j]
    output.to_csv(report_dir / 'test_predictions.csv', index=False)
    prob = predictions[selected]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    matrix = confusion_matrix(test.FTR, np.array(LABELS)[prob.argmax(axis=1)], labels=LABELS)
    axes[0].imshow(matrix, cmap='Blues')
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(matrix[i,j]), ha='center', va='center')
    axes[0].set(xticks=range(3), yticks=range(3), xticklabels=['Away','Draw','Home'], yticklabels=['Away','Draw','Home'], xlabel='Predicted', ylabel='Actual', title='2025/26 confusion matrix')
    for j, label in enumerate(LABELS):
        observed, predicted = calibration_curve((test.FTR == label).astype(int), prob[:,j], n_bins=5, strategy='quantile')
        axes[1].plot(predicted, observed, 'o-', label=label)
    axes[1].plot([0,1],[0,1], '--', color='gray')
    axes[1].set(xlabel='Predicted probability', ylabel='Observed frequency', title='Reliability by outcome (5 quantile bins)', xlim=(0,1), ylim=(0,1))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(report_dir / 'evaluation.png', dpi=160)
    plt.close(fig)
    table = '| Model | Accuracy | Log loss ↓ | Brier ↓ |\n|---|---:|---:|---:|\n'
    for name, metric in results.items():
        table += f"| {name} | {metric['accuracy']:.1%} | {metric['log_loss']:.4f} | {metric['brier_multiclass']:.4f} |\n"
    report = f'''# Evaluation report

Selected on validation log loss: **{selected}**. Test season: **2025/26** ({len(test)} matches).

{table}
## Protocol

- Warm-up: 2015/16. Training: 2016/17–2023/24 ({len(train)} matches).
- Model selection: 2024/25 ({len(val)} matches), with fixed candidate settings.
- Refit on training + validation, then evaluate on 2025/26. Test scores do not select the winner.
- Parameters stay frozen throughout the test season; form and Elo update after each completed match date. This is sequential pre-match evaluation, not a season-ahead forecast.
- Same-day fixtures share the state from before that date. No current-match goals, shots, odds or outcome enter features.
- Elo starts at 1500, K=20, home adjustment=60; ratings regress 25% toward 1500 each season. Form uses the last five games within that season; missing form uses fixed priors.
- Brier is the mean sum of squared errors across three classes (range 0–2). Log loss is the main probability metric. Lower is better for both.
- Calibration curves are descriptive and noisy for 380 matches. No probability calibration was fitted on the test data.

![Evaluation](evaluation.png)

## Limits

One held-out season is limited evidence. No injuries, lineups, transfers or bookmaker information are used. Promoted teams without Premier League history start at the default Elo. These are estimates, not guaranteed outcomes. Draw recall can be poor even when probability metrics improve. No profit or betting-performance claim is made.

The separate demo model is refitted on all eligible seasons only AFTER evaluation. Its in-sample outputs must not be presented as test performance. Demo state ends on {metadata['data_cutoff']}; it is a historical snapshot, not a live fixture forecast.

Source: [Football-Data](https://www.football-data.co.uk/englandm.php). Exact downloaded file hashes are in `data/manifest.json`.
'''
    (report_dir / 'evaluation.md').write_text(report)
    joblib.dump(evaluation_model, artifact_dir / 'evaluation_model.joblib')
    demo_model = candidates()[selected]
    demo_model.fit(eligible[FEATURES], eligible.FTR)
    joblib.dump(demo_model, artifact_dir / 'demo_model.joblib')
    (artifact_dir / 'state.json').write_text(json.dumps({'state': state.to_dict(), 'teams': sorted(set(test.HomeTeam) | set(test.AwayTeam)), 'metadata': metadata}, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
