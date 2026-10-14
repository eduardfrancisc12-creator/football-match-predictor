import pandas as pd
import numpy as np
import pytest
from football.features import build_features, FEATURES, FormState
from football.data import validate


def sample():
    return pd.DataFrame([
        ['2024-08-01','2425','A','B',2,0,'H'],
        ['2024-08-01','2425','C','D',1,1,'D'],
        ['2024-08-08','2425','B','C',0,1,'A'],
        ['2024-08-15','2425','A','D',0,0,'D'],
    ], columns=['Date','Season','HomeTeam','AwayTeam','FTHG','FTAG','FTR']).assign(Date=lambda d:pd.to_datetime(d.Date))


def test_no_current_or_future_result_leakage():
    original = sample()
    a, _ = build_features(original)
    changed = original.copy()
    changed.loc[2, ['FTHG','FTAG','FTR']] = [8,0,'H']
    b, _ = build_features(changed)
    pd.testing.assert_frame_equal(a.loc[:2,FEATURES], b.loc[:2,FEATURES])


def test_prefix_invariance():
    a, _ = build_features(sample())
    b, _ = build_features(sample().iloc[:3])
    pd.testing.assert_frame_equal(a.iloc[:3].reset_index(drop=True), b)


def test_previous_results_affect_next_fixture():
    frame, _ = build_features(sample())
    assert frame.iloc[0].elo_diff == 0
    assert frame.iloc[2].elo_diff < 0
    assert frame.iloc[2].home_history == 1


def test_same_day_order_invariance():
    a, _ = build_features(sample())
    b, _ = build_features(sample().iloc[[1,0,2,3]])
    pd.testing.assert_frame_equal(a,b)


def test_same_date_results_cannot_affect_other_rows():
    # Artificial double fixture stresses the date-batch boundary.
    df = sample()
    df.loc[1,'HomeTeam'] = 'A'
    a, _ = build_features(df)
    df.loc[0, ['FTHG','FTAG','FTR']] = [0,9,'A']
    b, _ = build_features(df)
    pd.testing.assert_frame_equal(a.loc[:1,FEATURES], b.loc[:1,FEATURES])


def test_season_reset_and_serialization():
    _, state = build_features(sample())
    restored = FormState.from_dict(state.to_dict())
    assert restored.features('A','B') == state.features('A','B')
    old = restored.elo['A']
    restored.new_season('2526')
    assert restored.elo['A'] == 1500 + .75*(old-1500)
    assert restored.features('A','B')['home_history'] == 0
    with pytest.raises(ValueError):
        state.features('A','A')


def test_bad_result_rejected():
    df = sample().drop(columns='Season')
    df.loc[0,'FTR'] = 'A'
    with pytest.raises(ValueError,match='result does not match'):
        validate(df,'2425',expected_matches=4)
