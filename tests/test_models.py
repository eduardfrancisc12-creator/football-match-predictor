import numpy as np
import pandas as pd
from football.features import FEATURES
from football.train import candidates, scores


def test_all_candidates_produce_valid_ordered_probabilities():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, len(FEATURES))), columns=FEATURES)
    frame['FTR'] = ['A','D','H'] * 40
    for model in candidates().values():
        model.fit(frame.iloc[:90][FEATURES], frame.iloc[:90].FTR)
        metrics, p = scores(model, frame.iloc[90:])
        assert p.shape == (30,3)
        assert np.all(p >= 0) and np.all(p <= 1)
        np.testing.assert_allclose(p.sum(axis=1), 1.)
        assert np.isfinite(list(metrics.values())).all()
