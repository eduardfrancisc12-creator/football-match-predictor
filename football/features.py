"""Features are emitted BEFORE any result from the same date updates state."""
from collections import defaultdict
import numpy as np
import pandas as pd

FEATURES = ['elo_diff', 'home_elo', 'away_elo', 'points_diff', 'scored_diff', 'conceded_diff', 'home_history', 'away_history']


class FormState:
    def __init__(self):
        self.elo = defaultdict(lambda: 1500.0)
        self.history = defaultdict(list)
        self.season = None

    def new_season(self, season):
        if self.season is not None and self.season != season:
            self.elo = defaultdict(lambda: 1500.0, {k: 1500 + .75 * (v - 1500) for k, v in self.elo.items()})
            self.history = defaultdict(list)
        self.season = season

    def summary(self, team):
        rows = self.history.get(team, [])[-5:]
        return np.mean(rows, axis=0) if rows else np.array([1.3, 1.35, 1.35])

    def features(self, home, away):
        if home == away:
            raise ValueError('Choose two different teams.')
        h, a = self.summary(home), self.summary(away)
        he, ae = self.elo.get(home, 1500.), self.elo.get(away, 1500.)
        return dict(zip(FEATURES, [he - ae, he, ae, h[0] - a[0], h[1] - a[1], h[2] - a[2], min(len(self.history.get(home, [])), 5), min(len(self.history.get(away, [])), 5)]))

    def update(self, row):
        home, away = row.HomeTeam, row.AwayTeam
        hg, ag = row.FTHG, row.FTAG
        result = 1. if hg > ag else 0. if hg < ag else .5
        expected = 1 / (1 + 10 ** ((self.elo[away] - self.elo[home] - 60) / 400))
        delta = 20 * (result - expected)
        self.elo[home] += delta
        self.elo[away] -= delta
        self.history[home] = (self.history[home] + [[3 if hg > ag else 1 if hg == ag else 0, hg, ag]])[-5:]
        self.history[away] = (self.history[away] + [[3 if ag > hg else 1 if hg == ag else 0, ag, hg]])[-5:]

    def to_dict(self):
        return {'elo': dict(self.elo), 'history': dict(self.history), 'season': self.season}

    @classmethod
    def from_dict(cls, payload):
        obj = cls()
        obj.elo.update(payload['elo'])
        obj.history.update(payload['history'])
        obj.season = payload['season']
        return obj


def build_features(matches):
    state = FormState()
    records = []
    for date, batch in matches.sort_values(['Date', 'HomeTeam']).groupby('Date', sort=True):
        if batch.Season.nunique() != 1:
            raise ValueError('Multiple seasons on the same date')
        state.new_season(batch.Season.iloc[0])
        for row in batch.itertuples():
            records.append({'Date': date, 'Season': row.Season, 'HomeTeam': row.HomeTeam, 'AwayTeam': row.AwayTeam, 'FTR': row.FTR, **state.features(row.HomeTeam, row.AwayTeam)})
        for row in batch.itertuples():
            state.update(row)
    return pd.DataFrame(records), state
