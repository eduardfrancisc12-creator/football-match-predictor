from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_app_or_setup_message():
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=30)
    assert not app.exception
    if not (ROOT / 'artifacts' / 'state.json').exists():
        assert 'Train the model first' in app.info[0].value
        return
    assert len(app.metric) == 3
    app.selectbox[0].select('Liverpool').run(timeout=30)
    assert not app.exception
    assert app.selectbox[0].value != app.selectbox[1].value
    probabilities = [float(m.value.rstrip('%')) for m in app.metric]
    assert sum(probabilities) == pytest.approx(100, abs=.2)
