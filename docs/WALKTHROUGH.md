# Understand your ML project

## 1. What is being learned?

Each row is one football match. The target is a category: H (home win), D (draw), A (away win). We want a probability for each category, not a certain winner. This is supervised multiclass classification.

## 2. Why not feed in the score?

The score determines the target and is unknown before kickoff. Feeding it into training would teach the model a shortcut unavailable at prediction time. The feature builder reads history first, emits the feature row, and only then updates history using the result.

Try changing a future result and rerun `test_no_current_or_future_result_leakage`. Earlier feature rows should remain identical. This is more useful than only checking that a function returns a DataFrame.

## 3. What does Elo add?

Elo is a running estimate of team strength. Beating a strong opponent increases a rating more than beating a weak one. The expected score uses a logistic curve of rating difference; updates use actual minus expected score. A draw is 0.5 in the Elo update, but the classifier still treats a draw as its own class.

## 4. Why three datasets?

Training fits parameters. Validation chooses among candidate models. Test estimates future generalization. Selecting the best test model would reuse the test set for selection and make the reported result optimistic. This project chooses by validation log loss before computing test scores.

## 5. Why a baseline?

A model can look impressive without beating a simple alternative. The baseline assigns the same outcome-frequency probabilities to every match. Improvements in log loss show whether team features help beyond those frequencies.

## 6. Why log loss?

It rewards assigning high probability to the outcome that actually occurs and penalizes confident mistakes. A forecast of 40% home / 30% draw / 30% away is meaningfully different from 95% / 3% / 2%, even if both choose a home win.

## 7. What does the demo mean?

It uses the final stored team state and a separately refitted model. Arsenal versus Liverpool is a hypothetical match under that historical snapshot. It does not know today's injuries or future schedules. The app explicitly shows the cutoff date.

## Interview explanation

"I built a reproducible multiclass classification pipeline for Premier League outcomes. I generated pre-match Elo and rolling-form features, compared a frequency baseline, logistic regression and gradient boosting, selected using chronological validation log loss, and evaluated on a later untouched season. I tested for result leakage and built a Streamlit interface. My main limitations are missing lineup information, promoted-team priors and evaluation on only one season."

Replace generalities with the actual metrics from `reports/evaluation.md`. Do not claim professional forecasting experience or training a large language model: this project trains tabular classifiers.

## Exercises to make it yours

1. Explain one row of features manually using the five previous fixtures.
2. Calculate the Elo update for a surprising away win.
3. Inspect test matches with the highest log loss and describe likely missing information.
4. Add a rest-days feature, with a test proving it uses only prior dates.
5. Evaluate that change with expanding-window validation; reserve a new season before making a new final performance claim.
