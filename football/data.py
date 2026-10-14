"""Download and validate completed Premier League seasons."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
SEASONS = [f'{y:02d}{y + 1:02d}' for y in range(15, 26)]
REQUIRED = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']


def validate(frame, season, expected_matches=380):
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f'{season}: missing columns {missing}')
    df = frame[REQUIRED].dropna(how='all').copy()
    if df.isna().any().any():
        raise ValueError(f'{season}: incomplete results')
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed', errors='raise')
    for col in ['FTHG', 'FTAG']:
        values = pd.to_numeric(df[col], errors='raise')
        if ((values < 0) | (values % 1 != 0)).any():
            raise ValueError(f'{season}: invalid goals')
        df[col] = values.astype(int)
    actual = df.apply(lambda r: 'H' if r.FTHG > r.FTAG else 'A' if r.FTHG < r.FTAG else 'D', axis=1)
    if not actual.equals(df.FTR):
        raise ValueError(f'{season}: result does not match score')
    if df.duplicated(['HomeTeam', 'AwayTeam']).any() or (df.HomeTeam == df.AwayTeam).any():
        raise ValueError(f'{season}: duplicate or invalid fixture')
    if len(df) != expected_matches:
        raise ValueError(f'{season}: expected {expected_matches} completed matches, got {len(df)}')
    start = 2000 + int(season[:2])
    # 2019/20 continued into July after the COVID interruption.
    if not df.Date.between(f'{start}-07-01', f'{start + 1}-08-31').all():
        raise ValueError(f'{season}: dates outside season')
    df['Season'] = season
    return df


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = []
    for season in SEASONS:
        path = RAW / f'{season}.csv'
        url = f'https://www.football-data.co.uk/mmz4281/{season}/E0.csv'
        if not path.exists():
            request = Request(url, headers={'User-Agent': 'FootballMatchPredictor/1.0 (educational portfolio)'})
            with urlopen(request, timeout=45) as response:
                content = response.read()
            import io
            validate(pd.read_csv(io.BytesIO(content)), season)
            path.write_bytes(content)
        validate(pd.read_csv(path), season)
        manifest.append({'season': season, 'source': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'matches': 380})
        print(f'{season}: validated 380 matches', flush=True)
    (ROOT / 'data' / 'manifest.json').write_text(json.dumps({'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'files': manifest}, indent=2))


def load_matches():
    return pd.concat([validate(pd.read_csv(RAW / f'{s}.csv'), s) for s in SEASONS], ignore_index=True).sort_values(['Date', 'HomeTeam']).reset_index(drop=True)


if __name__ == '__main__':
    download()
