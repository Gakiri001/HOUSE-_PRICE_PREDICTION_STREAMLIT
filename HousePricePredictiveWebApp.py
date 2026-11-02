"""Streamlit interface for the existing Boston-style house price model."""

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st

FEATURE_ORDER = [
    "crim",
    "zn",
    "indus",
    "chas",
    "nox",
    "rm",
    "age",
    "dis",
    "rad",
    "tax",
    "ptratio",
    "b",
    "lstat",
]
MODEL_PATH = Path(__file__).resolve().parent / "house_price_model.joblib"

st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f7fafc 0%, #eef4f8 100%); }
    .block-container { max-width: 1160px; padding-top: 2.2rem; padding-bottom: 3rem; }
    .hero { padding: 1.65rem 1.8rem; border-radius: 20px; color: white;
        background: linear-gradient(115deg, #123b52, #197b78); margin-bottom: 1.5rem; }
    .hero h1 { color: white; margin: 0 0 .45rem 0; font-size: clamp(1.8rem, 4vw, 2.65rem); }
    .hero p { color: #e3f4f2; margin: 0; font-size: 1.04rem; }
    div[data-testid="stForm"] { padding: 1.25rem; border: 1px solid #dce7eb;
        border-radius: 16px; background: rgba(255,255,255,.9); }
    div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
        width: 100%; min-height: 3rem; border: 0; border-radius: 10px;
        color: white; font-weight: 700; background: #147c78; }
    div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background: #105f5c; color: white; }
    .result { border: 1px solid #c9e6df; background: #f0faf6; border-radius: 16px;
        padding: 1.2rem 1.4rem; margin-top: 1.25rem; }
    @media (max-width: 640px) { .block-container { padding: 1rem .8rem 2rem; }
        .hero { padding: 1.2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained model…")
def load_model(path: str):
    """Load the saved model once per Streamlit process."""
    return joblib.load(path)


st.markdown(
    """
    <div class="hero">
      <h1>🏡 House Price Prediction System</h1>
      <p>Explore a machine learning estimate using property details and neighbourhood characteristics.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not MODEL_PATH.is_file():
    st.error(f"The trained model file was not found at `{MODEL_PATH}`.")
    st.stop()

try:
    saved_artifact = load_model(str(MODEL_PATH))
    # The artifact may be a plain estimator or a dictionary containing it.
    model = (
        saved_artifact.get("model", saved_artifact)
        if isinstance(saved_artifact, dict)
        else saved_artifact
    )
except Exception as exc:
    st.error(
        "The trained model could not be loaded. Install the project dependencies "
        "from `requirements.txt` and confirm the model file is valid. "
        f"Details: `{type(exc).__name__}: {exc}`"
    )
    st.stop()

st.subheader("Property and neighbourhood details")
st.caption(
    "Enter values for all 13 model features. Hover over each label for a definition."
)

with st.form("prediction_form"):
    property_col, location_col, environment_col = st.columns(3, gap="large")

    with property_col:
        st.markdown("#### 🏠 Property characteristics")
        rm = st.number_input(
            "Average rooms (RM)",
            min_value=1.0,
            max_value=20.0,
            value=6.0,
            step=0.1,
            help="Average number of rooms per dwelling.",
        )
        age = st.number_input(
            "Older homes (AGE, %)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
            help="Proportion of owner-occupied units built before 1940.",
        )
        lstat = st.number_input(
            "Lower-status population (LSTAT, %)",
            min_value=0.0,
            max_value=100.0,
            value=12.0,
            step=0.1,
            help="Percentage of lower-status population in the area.",
        )

    with location_col:
        st.markdown("#### 📍 Location and access")
        crim = st.number_input(
            "Crime rate (CRIM)",
            min_value=0.0,
            max_value=100.0,
            value=0.1,
            step=0.01,
            format="%.3f",
            help="Per capita crime rate by town.",
        )
        zn = st.number_input(
            "Large-lot zoning (ZN, %)",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="Residential land zoned for lots over 25,000 sq. ft.",
        )
        indus = st.number_input(
            "Non-retail business land (INDUS, %)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="Proportion of non-retail business acres per town.",
        )
        chas = st.selectbox(
            "Charles River tract (CHAS)",
            options=[0, 1],
            format_func=lambda value: "No (0)" if value == 0 else "Yes (1)",
            help="1 if the tract bounds the Charles River; otherwise 0.",
        )
        dis = st.number_input(
            "Employment-centre distance (DIS)",
            min_value=0.1,
            max_value=20.0,
            value=4.0,
            step=0.1,
            help="Weighted distance to five Boston employment centres.",
        )
        rad = st.number_input(
            "Highway access index (RAD)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=1.0,
            help="Index of accessibility to radial highways.",
        )

    with environment_col:
        st.markdown("#### 🌿 Environment and economy")
        nox = st.number_input(
            "Nitric oxide (NOX)",
            min_value=0.2,
            max_value=1.0,
            value=0.55,
            step=0.01,
            format="%.3f",
            help="Nitric oxide concentration.",
        )
        tax = st.number_input(
            "Property tax (TAX)",
            min_value=100.0,
            max_value=1000.0,
            value=300.0,
            step=1.0,
            help="Full-value property-tax rate per $10,000.",
        )
        ptratio = st.number_input(
            "Pupil–teacher ratio (PTRATIO)",
            min_value=5.0,
            max_value=30.0,
            value=18.0,
            step=0.1,
            help="Pupil-teacher ratio by town.",
        )
        b_value = st.number_input(
            "Neighbourhood B value (B)",
            min_value=0.0,
            max_value=400.0,
            value=390.0,
            step=1.0,
            help="1000(Bk − 0.63)², as defined in the original dataset.",
        )

    submitted = st.form_submit_button("Predict House Price", type="primary")

if submitted:
    values = [
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        dis,
        rad,
        tax,
        ptratio,
        b_value,
        lstat,
    ]
    if len(values) != len(FEATURE_ORDER) or not all(
        math.isfinite(float(value)) for value in values
    ):
        st.error("Please provide valid finite values for all 13 features.")
    else:
        features = pd.DataFrame([values], columns=FEATURE_ORDER)
        try:
            prediction = np.asarray(model.predict(features)).reshape(-1)
            if prediction.size != 1 or not math.isfinite(float(prediction[0])):
                raise ValueError("The model did not return one finite price estimate.")
            price = float(prediction[0])
            st.session_state["last_prediction"] = price
        except Exception as exc:
            st.error(
                f"Prediction failed. Check the entered feature values and model compatibility. Details: `{type(exc).__name__}: {exc}`"
            )

if "last_prediction" in st.session_state:
    st.markdown('<div class="result">', unsafe_allow_html=True)
    st.subheader("Predicted House Price")
    st.metric(
        "Estimated median value", f"${st.session_state['last_prediction']:.2f} thousand"
    )
    st.caption(
        "The saved model's training code is not included, so the displayed unit follows the Boston Housing convention: USD thousands. This is a machine learning estimate, not a guaranteed market valuation."
    )
    st.markdown("</div>", unsafe_allow_html=True)
