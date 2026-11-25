from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "models" / "dr_effnet_b0.pt").exists(), reason="trained model not present")


def test_app_renders_and_loads_the_model():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    assert at.title[0].value == "Diabetic retinopathy grading"
    assert not at.error                      # the model file was found and loaded
    assert len(at.sidebar.warning) == 1      # the "not a medical device" notice is shown
