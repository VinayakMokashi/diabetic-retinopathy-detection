"""Streamlit demo: grade a colour fundus photograph and show where the model looked.

Run from the repository root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dr import GRADES  # noqa: E402
from dr.inference import Predictor  # noqa: E402

MODEL_PATH = ROOT / "models" / "dr_effnet_b0.pt"
METRICS_PATH = ROOT / "reports" / "metrics.json"

DESCRIPTIONS = {
    0: "No visible signs of diabetic retinopathy.",
    1: "Microaneurysms only.",
    2: "More than microaneurysms, but less than severe NPDR (e.g. haemorrhages, hard exudates, cotton-wool spots).",
    3: "Extensive haemorrhages in all four quadrants, venous beading, or intraretinal microvascular abnormalities.",
    4: "Neovascularisation or vitreous/pre-retinal haemorrhage; laser scars from earlier treatment are also common.",
}


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor(MODEL_PATH)


def sidebar() -> None:
    st.sidebar.header("About the model")
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        t = m["test"]
        st.sidebar.markdown(
            f"**{m['arch'].split('.')[0]}** fine-tuned at {m['image_size']} px on 1,160 EyePACS images.\n\n"
            f"Held-out test set ({t['n_images']} images, {t['n_patients']} patients):\n"
            f"- quadratic weighted kappa **{t['quadratic_kappa']:.3f}**\n"
            f"- exact grade accuracy **{t['accuracy']:.1%}**\n"
            f"- referable DR (grade >= 2) AUC **{t['referable_auc']:.3f}**, "
            f"sensitivity {t['referable_sensitivity']:.1%}, specificity {t['referable_specificity']:.1%}"
        )
    st.sidebar.warning(
        "Research demo only. This is not a medical device and must not be used to diagnose, screen or treat anyone."
    )


def main() -> None:
    st.set_page_config(page_title="Diabetic retinopathy grading", page_icon="👁️", layout="wide")
    st.title("Diabetic retinopathy grading")
    st.write("Upload a colour fundus photograph. The model grades it on the 0-4 ICDR scale and highlights the "
             "regions that raised its severity estimate.")
    sidebar()

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}. Train one with scripts/train.py.")
        return
    predictor = load_predictor()

    uploaded = st.file_uploader("Fundus photograph", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"])
    if uploaded is None:
        st.info("No image yet. Any macula- or disc-centred colour fundus photograph works; the EyePACS images from "
                "the Kaggle Diabetic Retinopathy Detection competition are a good source.")
        return

    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Grading..."):
        prediction = predictor.predict(image)
        heatmap, _ = predictor.explain(image)

    c1, c2, c3 = st.columns(3)
    c1.image(image, caption="Uploaded image", width="stretch")
    c2.image(prediction.preprocessed, caption="Model input (cropped, local-colour normalised)",
             width="stretch")
    c3.image(heatmap, caption="Grad-CAM: regions that raised the severity score", width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Grade {prediction.grade}: {prediction.grade_name}")
        st.write(DESCRIPTIONS[prediction.grade])
        st.metric("Severity score (0-4)", f"{prediction.severity_score:.2f}")
        st.metric("Probability of referable DR (grade >= 2)", f"{prediction.referable_probability:.0%}")
        if prediction.referable:
            st.error("Referable: the model would send this patient to an eye specialist.")
        else:
            st.success("Not referable by the model's decision rule.")
    with right:
        probs = pd.DataFrame({"probability": prediction.probabilities},
                             index=[f"{g} {GRADES[g]}" for g in range(5)])
        st.bar_chart(probs, horizontal=True)


if __name__ == "__main__":
    main()
