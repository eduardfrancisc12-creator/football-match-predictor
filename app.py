"""Historical matchup explorer; run: streamlit run app.py."""
import json
import joblib
import pandas as pd
import streamlit as st
from football.data import ROOT
from football.features import FormState, FEATURES

st.set_page_config(page_title='Matchday Lab', page_icon='⚽', layout='wide')
st.title('⚽ Matchday Lab')
st.caption('Premier League · Match probabilities · An ML portfolio by Eduard Szekely')
path = ROOT / 'artifacts' / 'state.json'
if not path.exists():
    st.info('Train the model first: python -m football.data && python -m football.train')
    st.stop()

@st.cache_resource
def load_artifacts():
    payload = json.loads(path.read_text())
    return payload, joblib.load(ROOT / 'artifacts' / 'demo_model.joblib')

payload, model = load_artifacts()
meta = payload['metadata']
state = FormState.from_dict(payload['state'])
st.info(f"Historical snapshot through {meta['data_cutoff']}. This compares teams using their end-of-season form; it does not predict a scheduled or live match.")
left, right = st.columns(2)
teams = payload['teams']
home = left.selectbox('Home team', teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 0)
away_options = [team for team in teams if team != home]
away = right.selectbox('Away team', away_options, index=away_options.index('Liverpool') if 'Liverpool' in away_options else 0)
features = state.features(home, away)
probs = dict(zip(model.classes_, model.predict_proba(pd.DataFrame([features], columns=FEATURES))[0]))
for col, label, key in zip(st.columns(3), [f'{home} win', 'Draw', f'{away} win'], ['H','D','A']):
    col.metric(label, f'{probs[key]:.1%}')
st.bar_chart(pd.DataFrame({'Outcome': ['Home win','Draw','Away win'], 'Probability': [probs['H'],probs['D'],probs['A']]}).set_index('Outcome'))
st.subheader('What the model sees')
st.caption('Recent form means the last five league games of the snapshot season. These are inputs, not causal explanations.')
rows = []
for team in [home, away]:
    summary = state.summary(team)
    rows.append({'Team':team, 'Elo':round(state.elo.get(team,1500)), 'Points / game':round(summary[0],2), 'Goals scored / game':round(summary[1],2), 'Goals conceded / game':round(summary[2],2)})
st.dataframe(pd.DataFrame(rows).set_index('Team'), use_container_width=True)
with st.expander('How it was evaluated'):
    st.write(f"Selected model: {meta['selected_model']}. Selected using 2024/25 log loss, then tested on 380 matches in 2025/26 with frozen parameters and rolling historical features.")
    st.dataframe(pd.DataFrame(meta['test']).T, use_container_width=True)
    st.image(str(ROOT / 'reports' / 'evaluation.png'))
    st.caption('The demo uses a separate model refitted after testing. Test scores above belong to the evaluation models, not this refitted artifact.')
st.caption('Educational experiment. No lineups, injury information or live updates. Source: football-data.co.uk.')
