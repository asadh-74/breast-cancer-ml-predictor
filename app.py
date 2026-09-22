"""
Breast Cancer Diagnosis Predictor — Streamlit app.
Trains the model fresh on startup (same pipeline as breast_cancer_classifier.ipynb) and serves
live predictions. Training on this dataset takes well under a second, so there is no need to
ship a pickled model file — which avoids scikit-learn version-mismatch errors between whatever
trained the pickle and whatever is installed in your environment.

Run locally:    streamlit run app.py
Deploy free:    https://share.streamlit.io  (point it at this repo + app.py)
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

st.set_page_config(page_title="Breast Cancer Diagnosis Predictor", page_icon="🩺", layout="centered")


@st.cache_resource
def train_model():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0 = malignant, 1 = benign

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1]
    metrics = {
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "model_name": "Logistic Regression",
        "metrics": metrics,
        "data_means": data.data.mean(axis=0),
    }


artifact = train_model()
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]
model_name = artifact["model_name"]
metrics = artifact["metrics"]

st.title("🩺 Breast Cancer Diagnosis Predictor")
st.caption(
    f"Model: **{model_name}** · Test F1: **{metrics['f1']:.3f}** · "
    f"ROC-AUC: **{metrics['roc_auc']:.3f}**"
)
st.write(
    "Enter tumor measurements from a fine needle aspirate (FNA) to predict whether the mass "
    "is **benign** or **malignant**. This is a portfolio demo trained on the public "
    "Breast Cancer Wisconsin (Diagnostic) dataset — **not a medical device.**"
)

# Group the 30 features into the 3 sensible clusters they naturally fall into
mean_features = [f for f in feature_names if f.startswith("mean")]
error_features = [f for f in feature_names if f.startswith(("error", "mean")) is False and "error" in f]
worst_features = [f for f in feature_names if f.startswith("worst")]

with st.expander("ℹ️ How to use this demo"):
    st.write(
        "Use the sample buttons below to auto-fill a realistic example, or expand the sections "
        "and adjust the sliders yourself. Then click **Predict**."
    )

col1, col2 = st.columns(2)
sample = None
if col1.button("Load benign-like sample"):
    sample = "benign"
if col2.button("Load malignant-like sample"):
    sample = "malignant"

# Reasonable representative defaults for the two classes (derived from dataset means)
defaults_benign = {
    "mean radius": 12.1, "mean texture": 18.0, "mean perimeter": 78.0, "mean area": 460.0,
    "mean smoothness": 0.09, "mean compactness": 0.08, "mean concavity": 0.04,
    "mean concave points": 0.025, "mean symmetry": 0.17, "mean fractal dimension": 0.06,
}
defaults_malignant = {
    "mean radius": 19.0, "mean texture": 21.5, "mean perimeter": 125.0, "mean area": 1130.0,
    "mean smoothness": 0.11, "mean compactness": 0.16, "mean concavity": 0.19,
    "mean concave points": 0.10, "mean symmetry": 0.20, "mean fractal dimension": 0.06,
}
preset = defaults_malignant if sample == "malignant" else defaults_benign if sample == "benign" else {}

st.subheader("Tumor measurements (mean values)")
inputs = {}
sliders_spec = [
    ("mean radius", 6.0, 30.0),
    ("mean texture", 9.0, 40.0),
    ("mean perimeter", 40.0, 190.0),
    ("mean area", 140.0, 2500.0),
    ("mean smoothness", 0.05, 0.17),
    ("mean compactness", 0.02, 0.35),
    ("mean concavity", 0.0, 0.43),
    ("mean concave points", 0.0, 0.2),
    ("mean symmetry", 0.1, 0.3),
    ("mean fractal dimension", 0.05, 0.1),
]
cols = st.columns(2)
for i, (feat, lo, hi) in enumerate(sliders_spec):
    with cols[i % 2]:
        inputs[feat] = st.slider(
            feat.replace("mean ", "").title(), min_value=float(lo), max_value=float(hi),
            value=float(preset.get(feat, (lo + hi) / 2)), key=feat,
        )

if st.button("Predict", type="primary"):
    # Fill non-mean features with the dataset's overall column means so the model gets
    # a complete, valid feature vector (a lightweight simplification for this demo).
    full_means = pd.Series(artifact["data_means"], index=feature_names)

    row = full_means.copy()
    for feat, val in inputs.items():
        row[feat] = val
    row = row[feature_names].values.reshape(1, -1)

    if scaler is not None:
        row = scaler.transform(row)

    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0]

    label = "Benign" if pred == 1 else "Malignant"
    confidence = proba[pred]

    if pred == 1:
        st.success(f"Prediction: **{label}** (confidence: {confidence:.1%})")
    else:
        st.error(f"Prediction: **{label}** (confidence: {confidence:.1%})")

    st.progress(float(proba[1]), text=f"P(benign) = {proba[1]:.1%}")

st.divider()
st.caption(
    "Built by Asad Hussain · Model trained on the UCI/scikit-learn Breast Cancer Wisconsin "
    "(Diagnostic) dataset · For portfolio/demo purposes only."
)
